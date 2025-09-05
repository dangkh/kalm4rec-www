import torch
import os
from unsloth import FastLanguageModel
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import string
import shutil
import argparse
from reRanker.utils import *
from retrievalHelper.utils import *
import torch.nn.functional as F
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import Dataset
from datasets import load_dataset

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # tránh num_proc=68
device_map = {"": local_rank}


alpaca_prompt_tunning = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a restaurant recommender system. Given the keywords representing both the user and the restaurants, where each restaurant is identified by a letter in a multiple-choice list, your task is:
First, Rerank the restaurants (by their letters) based on how semantically relevant and suitable their keywords are to the user’s preferences, rather than simply matching identical words. Consider the meaning and context of the keywords to determine suitability. Focus on the top 5 most suitable restaurants.
Then, respond with a single uppercase letter representing the most suitable restaurant. After that, add a section titled `### Note` containing a list of all possible restaurants (letters).

### Input:
These are the keywords that user often mention when wanting to choose restaurants: {}.
The restaurant with the associated keywords have the following form: A: (keyword 1, keyword 2,...) are: \n
{}

### Response:
The most suitable restaurant is {}."""

auxilliary = """
### NOTE: All possible Restaurants:
{}
"""

mcn_prompt_tunning = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a restaurant recommender system. Given the keywords representing both the user and the restaurants, where each restaurant is identified by a letter in a multiple-choice list, your task is:
First, Rerank the restaurants (by their letters) based on how semantically relevant and suitable their keywords are to the user’s preferences, rather than simply matching identical words. Consider the meaning and context of the keywords to determine suitability. Focus on the top 5 most suitable restaurants.
Then, respond with a single uppercase letter representing the most suitable restaurant. 

### Input:
These are the keywords that user often mention when wanting to choose restaurants: {}.
The restaurant with the associated keywords have the following form: A: (keyword 1, keyword 2,...) are: \n
{}

### Response:
The most suitable restaurant is {}."""


list_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a restaurant recommender system. Given the keywords representing both the user and the restaurants, your task is:
First, Rerank the restaurants based on how semantically relevant and suitable their keywords are to the user’s preferences, rather than simply matching identical words. Consider the meaning and context of the keywords to determine suitability. Focus on the top 5 most suitable restaurants.
Then, respond with a list of all re-ranked restaurants, ordered from most to least suitable. No explaination.


### Input:
These are the keywords that user often mention when wanting to choose restaurants: {}.
Candidate restaurants for user are (format: [restaurant_id_1, restaurant_id_2, ...]): {}
The restaurant with the associated keywords have the following form: restaurant_id_1: (keyword 1, keyword 2,...) are: \n
{}

Provide TOP 15 most suitable restaurants from the candidate set, ordered from most to least suitable.
### Response:
{}"""



def formatting_prompts_func(data):
    return {"text": alpaca_prompt_tunning.format(data["user"], data["input"], data["top"]) + auxilliary.format(data["output"]) + EOS_TOKEN}

def formatting_MCNprompts_func(data):
    return {"text": mcn_prompt_tunning.format(data["user"], data["input"], data["top"]) + EOS_TOKEN}

def formatting_list_prompts_func(data):
    return {"text": list_prompt.format(data["user"], data["candidate"], data["input"],  data["output"]) + EOS_TOKEN}

if __name__ == '__main__':
    listcity = ['edinburgh', 'london', 'singapore', 'tripAdvisor', 'amazonBaby', 'amazonVideo']
    parser = argparse.ArgumentParser('infer Kalm4Rec')
    parser.add_argument('--city', type=str, default='singapore', help=f'choose city{listcity}')
    parser.add_argument('--pretrainName', type=str, default='None', help='name of pretrained model')
    parser.add_argument('--type', type=str, default='mct', help=f'mct: multiple choice + token, mcn: noNote, list')
    parser.add_argument('--LLM', type=str, default='LLama', help='LLama, Gemma')
    args = parser.parse_args()


    # load and print baseline:
    root_dir = 'reRanker/'

    ### Load data
    city = args.city
    data_user_test = read_json(f"data/out2LLMs/{args.city}_knn2rest.json")
    rest_kws = read_json(f"data/score/{args.city}-keywords-TFIUF.json")
    # user_kws_train = read_json(f'data/out2LLMs/{args.city}_user2candidate.json') #user-train


    ## review filesS
    if city == 'tripAdvisor':
        is_tripAdvisor = True
    else: 
        is_tripAdvisor= False
    gt_file = 'data/reviews/{}.csv'.format(args.city)
    gt, u2rs, map_rest_id2int = prepare_user2rests(gt_file, is_tripAdvisor = is_tripAdvisor)
    train_res_kw = get_kw_for_rest(rest_kws, map_rest_id2int)
    
    model_name = "unsloth/Meta-Llama-3.1-8B"
    if  args.LLM == "Gemma":
        model_name = "unsloth/gemma-2-9b"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 4096,
        dtype = torch.bfloat16,
        load_in_4bit = True,
        device_map = device_map,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    EOS_TOKEN = tokenizer.eos_token
    
    loadName = f"./data/out2LLMs/train_data_{city}.json"
    if args.type != 'mct':
        loadName = f"./data/out2LLMs/train_data_{city}_{args.type}.json"
    dataset = load_dataset("json", data_files= loadName, split= 'train')
    afterSent = "### Response:\nThe most suitable restaurant is"
    if args.type == 'mct':
        restaurantDataset = dataset.map(formatting_prompts_func)
    elif args.type == 'list':
        restaurantDataset = dataset.map(formatting_list_prompts_func)
        afterSent = "### Response:\n"
    else:
        restaurantDataset = dataset.map(formatting_MCNprompts_func)

    print(restaurantDataset[0]['text'])


    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = restaurantDataset,
        dataset_text_field = "text",
        response_template= afterSent,  # << gồm cả khoảng trắng cuối
        train_on_prompt=False,      # << chỉ tính loss sau response_template
        max_seq_length = 4096,
        dataset_num_proc = 0,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            num_train_epochs = 1, # Set this for 1 full training run.
            # max_steps = 1,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_torch",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use this for WandB etc
            gradient_checkpointing=False,
            ddp_find_unused_parameters=False,
        ),
    )

    trainer_stats = trainer.train()
    saveName = f"{city}_tunModel"
    if args.LLM == "Gemma":
        saveName = f"Gemma_{city}_tunModel"
    if args.type != "mct":
        saveName += args.type

    model.save_pretrained(saveName)
    tokenizer.save_pretrained(saveName)

    print("*"*50)
    print("*"*6, f"Saved tp {saveName}", "*"*6)
    print("*"*50)

