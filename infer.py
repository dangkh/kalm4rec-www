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
import re

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    You are a restaurant recommender system. Given the keywords representing both the user and the restaurants, where each restaurant is identified by a letter (A, B,...,T), your task is:
    First, Rerank the restaurants based on how semantically relevant and suitable their keywords are to the user’s preferences, rather than simply matching identical words. Consider the meaning and context of the keywords to determine suitability. Focus on the top 5 most suitable restaurants.
    Then, respond with a single uppercase letter representing the most suitable restaurant. After that, add a section titled `### Note` containing a list of all possible restaurants (letters), ordered from most to least suitable.


    ### Input:
    These are the keywords that user often mention when wanting to choose restaurants: {}.
    The restaurant with the associated keywords have the following form: A: (keyword 1, keyword 2,...) are: \n
    {}

    ### Response:
    The most suitable restaurant is"""

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
"""

def predict_answer(model, input_prompt):
    inputs = tokenizer([input_prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model(**inputs)

    # Get logits for the next token
    logits = output.logits[:, -1, :]

    probs = F.softmax(logits, dim=-1)

    # Letters A, B, ...
    letters = [chr(i) for i in range(ord('A'), ord('A') + 20)]

    token_ids = []
    for letter in letters:
        tokenized = tokenizer(" " + letter, add_special_tokens=False)["input_ids"]
        if len(tokenized) == 1:
            token_ids.append(tokenized[0])
        else:
            token_ids.append(tokenized[0])
    token_probs = {}
    for letter, tid in zip(letters, token_ids):
        if tid is not None:
            token_probs[letter] = probs[0, tid].item()

    # Sort and take top 15
    top_tokens = sorted(token_probs.items(), key=lambda x: x[1], reverse=True)[:15]

    res = [x for x, v in top_tokens]
    return res

def get_answerList(model, input_prompt):
    inputs = tokenizer([input_prompt], return_tensors = "pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens = 64, do_sample=False, num_beams=1, use_cache = False)
    gen_ids = outputs[:, inputs["input_ids"].shape[-1]:]
    preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    output = preds[0].strip()
    return output

if __name__ == '__main__':
    listcity = ['edinburgh', 'london', 'singapore', 'tripAdvisor', 'amazonBaby', 'amazonVideo']
    parser = argparse.ArgumentParser('infer Kalm4Rec')
    parser.add_argument('--city', type=str, default='singapore', help=f'choose city{listcity}')
    parser.add_argument('--type', type=str, default='mct', help=f'mct: multiple choice + token, mcl: multiple choice + list, list')
    parser.add_argument('--LLM', type=str, default='LLama', help='LLama, Gemma')
    parser.add_argument('--baseline', type=bool, default=False, help='print baseline')
    parser.add_argument('--use_tuning', type=bool, default=False, help='use pretrained or use tunModel')
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
    if args.baseline:
        print('baseline')
        print(city)
        user_dict = {}
        for uid in data_user_test.keys():
            user_dict[uid] = [map_rest_id2int[can] for can in data_user_test[uid]['candidate']]
        evalAll(user_dict, u2rs)


    if args.use_tuning:
        model_name = f"{city}_tunModel"
        if args.LLM == "Gemma":
            model_name = f"Gemma_{city}_tunModel"
        if args.type == "list":
            model_name += 'list'
    else:
        model_name = "unsloth/Meta-Llama-3.1-8B"
        if  args.LLM == "Gemma":
            model_name = "unsloth/gemma-2-9b"

    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = 4096,
            dtype = None,
            load_in_4bit = None,
        )
    FastLanguageModel.for_inference(model)
    print("*"*50)
    print("*"*8, f"Using {model_name}", "*"*8)
    print("*"*50)
    for kws_for_user in [4, 5]:
        for kws_for_rest in [5, 6, 8, 10, 15]:
            user_rank = dict()
            print("\n")
            print(f'kws_for_user: {kws_for_user}, kws_for_rest: {kws_for_rest} \n')
            for uid in tqdm(data_user_test.keys(), total=len(data_user_test)):
                user_kw = data_user_test[uid]['kw'][:kws_for_user]
                if args.type == "mct":
                    tmp_str, choices, tmp_str2 = cand_kw_fnMCT(uid, train_res_kw, data_user_test, map_rest_id2int, 20, kws_for_rest)
                    input_prompt = alpaca_prompt.format(', '.join(user_kw), tmp_str)
                    predicted_answer = predict_answer(model, input_prompt)
                    candidate = data_user_test[uid]['candidate']
                    answer = [candidate[ord(x)-ord('A')] for x in predicted_answer]
                    answer = [map_rest_id2int[can] for can in answer]
                elif args.type == "list":
                    candilist, tmp_str = cand_kw_fn_list(uid, train_res_kw, data_user_test, map_rest_id2int, 20, kws_for_rest)
                    input_prompt = list_prompt.format(', '.join(user_kw), candilist, tmp_str)
                    output = get_answerList(model, input_prompt)
                    first_line = output.strip().split("\n")[0]
                    output = [int(x) for x in re.findall(r"\d+", first_line)]
                    answer = [int(outX) for outX in output]
                else:
                    pass
                user_rank[uid] = answer

            evalAll(user_rank, u2rs)
    
