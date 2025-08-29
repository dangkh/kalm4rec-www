import pandas as pd
import json
from pprint import pprint
from google import genai
import json 
import os 
from tqdm import tqdm 
import time 
import numpy as np
from utils import *
import random
import argparse
import time
import socket
import httpx
import ast
import re
import tiktoken
import yaml

seed = 12
random.seed(seed)

sleep_int = 100
sleep_time = 10

def prompt(client, type_llm, inputPrompt):
    if type_llm == 'gemini_pro':
        generation_config = {'candidate_count':1, "max_output_tokens": 2048, "temperature": 0, "top_p": 0.99, "top_k": 32}
        responses = client.models.generate_content(
            model="models/gemini-1.5-pro",
            contents = inputPrompt,
            config=generation_config
        )
        print(responses.text)
        return responses.text
    return inputPrompt
    # elif type_llm =='chatGPT':
    #     response = client.chat.completions.create(
    #         model="GPT-35-turbo-WebQSP-Vanilla", # model = "deployment_name"
    #         messages=[
    #             {"role": "user", "content": inputPrompt}
    #         ],
    #         temperature=0,
    #         max_tokens=800,
    #         top_p=0.95,
    #         frequency_penalty=0,
    #         presence_penalty=0,
    #         stop=None
    #     )
    #     return response.choices[0].message.content


### FewShot
def fewshot(model, city, type_method, type_llm, data_user_test, user_kws_train, map_rest_id2int, new_results_res_kw, num_kws_user, num_kws_rest, samples, label_samples, root_dir):

    temp_fewshot = all_prompts[type_llm][type_method]
    user_rank = dict()
    counterRequest = 0
    len_rank = None
    folder_path_result_rerank = f'{root_dir}results_rerank/{city}'
    if not os.path.exists(folder_path_result_rerank):
        os.makedirs(folder_path_result_rerank)
    file_path_ = folder_path_result_rerank+ f"/{type_method}_{num_kws_user}_{num_kws_rest}.json"
    print(file_path_)
    if os.path.isfile(file_path_):
        print("File exists")
        with open(file_path_, "r") as file:
            user_rank = json.load(file)
        len_rank = len(user_rank.keys())
    for uid in tqdm(data_user_test.keys(), total=len(data_user_test)):
        if len_rank is not None:
            if counterRequest <= len_rank:
                counterRequest += 1 
                continue
        if (counterRequest+1) % sleep_int == 0:
            time.sleep(sleep_time)
        user_kw = data_user_test[uid]['kw'][: num_kws_user]  # use 5 kws for user
        res_candidate = list(map(int,[str(map_rest_id2int[cand]) for cand in data_user_test[uid]['candidate']]))
        ### selecting randomly users for examples
        userKeys = list(user_kws_train.keys())
        user_train = random.choice(userKeys)
        user_train_2 = random.choice(userKeys)
        user_train_3 = random.choice(userKeys)

        user_train_kw = list(user_kws_train[user_train])[: num_kws_user]
        candidate_train = samples[user_train]
        labels = label_samples[user_train]
        res_candidate_train = list(map(int,[str(map_rest_id2int[cand]) for cand in candidate_train]))
        label_res = list(map(int,[str(map_rest_id2int[cand]) for cand in labels]))

        if type_method == '2_shots':
            user_train_kw_2 = list(user_kws_train[user_train_2])[: num_kws_user]
            # random.shuffle(user_train_kw_2)
            candidate_train_2 = samples[user_train_2]
            labels_2 = label_samples[user_train_2]
            res_candidate_train_2 = list(map(int,[str(map_rest_id2int[cand]) for cand in candidate_train_2]))
            label_res_2 = list(map(int,[str(map_rest_id2int[cand]) for cand in labels_2]))
        if type_method == '3_shots':
            user_train_kw_2 = list(user_kws_train[user_train_2])[: num_kws_user]
            # random.shuffle(user_train_kw_2)
            candidate_train_2 = samples[user_train_2]
            labels_2 = label_samples[user_train_2]
            res_candidate_train_2 = list(map(int,[str(map_rest_id2int[cand]) for cand in candidate_train_2]))
            label_res_2 = list(map(int,[str(map_rest_id2int[cand]) for cand in labels_2]))

            user_train_kw_3 = list(user_kws_train[user_train_3])[: num_kws_user]
            # random.shuffle(user_train_kw_3)
            candidate_train_3 = samples[user_train_3]
            labels_3 = label_samples[user_train_3]
            res_candidate_train_3 = list(map(int,[str(map_rest_id2int[cand]) for cand in candidate_train_3]))
            label_res_3 = list(map(int,[str(map_rest_id2int[cand]) for cand in labels_3]))

        if type_method == '1_shot':
            inputPrompt = temp_fewshot.format(', '.join(user_train_kw), res_candidate_train , cand_kw_fn_fewshot(user_train, new_results_res_kw,samples, map_rest_id2int, 20, num_kws_rest),label_res,
                                     ', '.join(user_kw),res_candidate, cand_kw_fn(uid, new_results_res_kw,data_user_test, map_rest_id2int, 20, num_kws_rest))
        elif type_method == '2_shots':
            inputPrompt = temp_fewshot.format(', '.join(user_train_kw), res_candidate_train , cand_kw_fn_fewshot(user_train, new_results_res_kw,samples, map_rest_id2int, 20, num_kws_rest),label_res,
                                        ', '.join(user_train_kw_2), res_candidate_train_2 , cand_kw_fn_fewshot(user_train_2, new_results_res_kw,samples, map_rest_id2int, 20, num_kws_rest),label_res_2,
                                     ', '.join(user_kw),res_candidate, cand_kw_fn(uid, new_results_res_kw,data_user_test, map_rest_id2int, 20, num_kws_rest))
        elif type_method == '3_shots':
            inputPrompt = temp_fewshot.format(', '.join(user_train_kw), res_candidate_train , cand_kw_fn_fewshot(user_train, new_results_res_kw,samples, map_rest_id2int, 20, num_kws_rest),label_res,
                                        ', '.join(user_train_kw_2), res_candidate_train_2 , cand_kw_fn_fewshot(user_train_2, new_results_res_kw,samples, map_rest_id2int, 20, num_kws_rest),label_res_2,
                                        ', '.join(user_train_kw_3), res_candidate_train_3 , cand_kw_fn_fewshot(user_train_3, new_results_res_kw,samples, map_rest_id2int, 20, num_kws_rest),label_res_3,
                                     ', '.join(user_kw),res_candidate, cand_kw_fn(uid, new_results_res_kw,data_user_test, map_rest_id2int, 20, num_kws_rest))

        predictions = prompt(model,type_llm, inputPrompt)
               
        pred = predictions.split(',')
        user_rank[uid] = list(map(int, pred))
        with open(file_path_, "w") as json_file:
            json.dump(user_rank, json_file)
    return user_rank


def extract_numbers(input_string):
    numbers = ''
    for char in input_string:
        if char.isdigit():
            numbers += char
    return numbers


def zeroshot(model, type_llm, data_user_test,  map_rest_id2int, new_results_res_kw, num_kws_user, num_kws_rest, root_dir):
    if city == 'tripAdvisor':
        temp = '''
        There are the keywords that I often mention when wanting to choose hotels: {}.
        The candidate hotel set for me is enclosed in square brackets, with the hotels separated by commas (Format: [hotel_id_1, hotel_id_2,...]) is: {}
        Keywords associated with candidate hotels are in the following format: hotel_id_1 (keyword 1, keyword 2,...) are {}.
        Input: Please suggest the 15 most suitable hotels for me from the candidate set that I will choose to stay, according to the user and candidate hotel keywords I provided above.
        Output: Must include 15 hotels in the candidate hotel set. No explanation. Desired format is string: hotel_id_1, hotel_id_2, ... 
        '''
    else:
        temp = '''
        There are the keywords that I often mention when wanting to choose items on Amazon: {}.
        The candidate items set for me is enclosed in square brackets, with the items separated by commas (Format: [item_id_1, item_id_2,...]) is: {}
        Keywords associated with candidate items have the following form: item_id_1 (keyword 1, keyword 2,...) are {}.
        Input: Please suggest the 15 most suitable items for me from the candidate set that I will purchase them, according to the user and candidate item keywords I provided above.
        Output: Must include 15 items in the candidate item set. No explanation. Desired format is string: item_id_1, item_id_2, ... 
        '''

        # temp = '''
        # There are the keywords that I often mention when wanting to choose restaurants: {}.
        # The candidate restaurant set for me is enclosed in square brackets, with the restaurants separated by commas (Format: [restaurant_id_1, restaurant_id_2,...]) is: {}
        # Keywords associated with candidate restaurants have the following form: restaurant_id_1 (keyword 1, keyword 2,...) are {}.
        # Input: Please suggest the 15 most suitable restaurants for me from the candidate set that I will visit them, according to the user and candidate restaurant keywords I provided above.
        # Output: Must include 15 restaurants in the candidate restaurant set. No explanation. Desired format is string: restaurant_id_1, restaurant_id_2, ... 
        # '''

    user_rank = dict()
    # user_shuffle = dict()
    len_rank = None
    folder_path_result_rerank = f'{root_dir}results_rerank/{city}'
    if not os.path.exists(folder_path_result_rerank):
        os.makedirs(folder_path_result_rerank)
    file_path_ = folder_path_result_rerank+ f"/zeroshot_{num_kws_user}_{num_kws_rest}_{seed}.json"
    # file_shuffle = folder_path_result_rerank+ f"/shuffle_cadidate_zeroshot_{num_kws_user}_{num_kws_rest}_{seed}.json"
    if os.path.isfile(file_path_):
        print("File exists")
        with open(file_path_, "r") as file:
            user_rank = json.load(file)
        len_rank = len(user_rank.keys())
    counterRequest = 0
    for uid in tqdm(data_user_test.keys(), total=len(data_user_test)):
        counterRequest = counterRequest+1
        if counterRequest %1000 == 0:
            time.sleep(120)
        if len_rank is not None:
            if counterRequest <= len_rank:
                counterRequest += 1
                continue
        user_kw = data_user_test[uid]['kw'][:num_kws_user] 
        res_candidate = list(map(int,[str(map_rest_id2int[cand]) for cand in data_user_test[uid]['candidate']])) 
        
        # ablation random
        # random.shuffle(res_candidate)
        # user_shuffle[uid] = res_candidate
        
        inputPrompt = temp.format(', '.join(user_kw),res_candidate, cand_kw_fn(uid, new_results_res_kw, data_user_test, map_rest_id2int, 20, num_kws_rest))
        flag = False
        while flag is False:
            try:
                predictions = prompt(model, type_llm, inputPrompt)
                print(predictions)
                flag = True                
                pred_ = predictions.split(',')
                pred = []
                for aa in pred_:
                    pred.append(extract_numbers(aa))
            except httpx.ReadTimeout:
                print("Timeout occurred. Connection timed out.")
                time.sleep(120)    
            except RuntimeError as e:
                print(f"{e}")
                time.sleep(120) 

        user_rank[uid] = list(map(int, pred))
        with open(file_path_, "w") as json_file:
            json.dump(user_rank, json_file)
        # with open(file_shuffle, "w") as json_file:
        #     json.dump(user_shuffle, json_file)

    return user_rank


if __name__ == '__main__':
    listcity = ['edinburgh', 'london', 'singapore']
    parser = argparse.ArgumentParser('LLM re-ranking RecSys')
    parser.add_argument('--type_method', type=str, default= 'zeroshot', help='zeroshot, 1_shot, 2_shots, 3_shots')
    parser.add_argument('--num_kws_user', type=int, default= 3)
    parser.add_argument('--num_kws_rest', type=int, default= 5)
    parser.add_argument('--city', type=str, default='singapore', help=f'choose city{listcity}')
    parser.add_argument('--type_LLM', type=str, default='gemini_pro', help='gemini_pro, chatGPT')
    parser.add_argument('--api_key', type=str, default=None, help='API key')


    args = parser.parse_args()
    root_dir = 'reRanker/'

    # if few-shots
    with open(root_dir+"prompts.yaml", "r") as f:
        all_prompts = yaml.safe_load(f)

    run_list_kws_for_user = [12, 10]
    run_list_kws_for_rest = [12, 10, 8]
    list_method = ['3_shots']

    ### Load data
    city = args.city
    data_user_test = read_json(f"data/out2LLMs/{args.city}_knn2rest.json")
    rest_kws = read_json(f"data/score/{args.city}-keywords-TFIUF.json")
    user_kws_train = read_json(f'data/out2LLMs/{args.city}_user2candidate.json') #user-train

    
    ## review filesS
    if city == 'tripAdvisor':
        is_tripAdvisor = True
    else: 
        is_tripAdvisor= False
    gt_file = 'data/reviews/{}.csv'.format(args.city)
    gt, u2rs, map_rest_id2int = prepare_user2rests(gt_file, is_tripAdvisor = is_tripAdvisor)

    new_results_res_kw = get_kw_for_rest(rest_kws, map_rest_id2int)

    # # sample for fewshot
    if '1_shot' in list_method or '2_shots' in list_method or '3_shots'in list_method:
        samples = read_json(f'data/fewshot_samples/{city}_5.json')
        label_samples = read_json(f'data/fewshot_samples/{city}_label_5.json')

    folder_path = root_dir + 'results'
    if not os.path.exists(folder_path):
        # If it doesn't exist, create it
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    file_path = f'{folder_path}/{args.city}_{seed}.txt'
    file_path_json = f'{folder_path}/{args.city}_{seed}.json'

    print('Begin run LLM tests')
    result_json = dict()
    for met in list_method:
        result_json[met] = dict()
        for num_kws_user in run_list_kws_for_user:
            result_json[met][num_kws_user] = dict()
            for num_kws_rest in run_list_kws_for_rest:
                result_json[met][num_kws_user][num_kws_rest] = dict()
                print('\nargs: ', args)
                print(f"Method: {met}, number of user keyword: {num_kws_user}, number of rest keyword: {num_kws_rest}")
                data_args = vars(args)
                with open(file_path, 'a') as file:
                # Write log
                    file.write("\n")
                    json.dump(data_args, file)
                    file.write("\n")
                    file.write(f"Method: {met}, number of user keyword: {num_kws_user}, number of rest keyword: {num_kws_rest}")
                    file.write("\n")

                if args.type_LLM == 'gemini_pro':
                    client = genai.Client(api_key="AIzaSyAr_JH8EFd27kXVm3tBwbHbYOqDkMpLgps")
                    if met in ['1_shot', '2_shots', '3_shots']:
                        user_rank = fewshot(client, city, met, args.type_LLM, data_user_test, user_kws_train, map_rest_id2int, new_results_res_kw, num_kws_user, num_kws_rest, samples, label_samples, root_dir)
                    elif met == 'zeroshot':
                        user_rank = zeroshot(client, args.type_LLM, data_user_test, map_rest_id2int, new_results_res_kw, num_kws_user, num_kws_rest, root_dir)                                        
                pre, rec, f1, ndcg = evalAll(user_rank, u2rs)
                result_json[met][num_kws_user][num_kws_rest]['prec'] = pre
                result_json[met][num_kws_user][num_kws_rest]['recall'] = rec
                result_json[met][num_kws_user][num_kws_rest]['f1'] = f1
                result_json[met][num_kws_user][num_kws_rest]['ndcg'] = ndcg

                with open(file_path_json, "a") as json_file:
                    json.dump(result_json, json_file)
                with open(file_path, 'a') as file:
                    file.write(str(result_json))
                    file.write("\n")
                print('slepping ...')