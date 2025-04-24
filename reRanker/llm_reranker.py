import pandas as pd
import json
from pprint import pprint
import google.generativeai as genai
import json 
import os 
from tqdm import tqdm 
import time 
import numpy as np
from utils import *
import random
import argparse
import google.generativeai as palm
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time
import socket
import httpx
import ast
import re
import tiktoken

seed = 12
random.seed(seed)

sleep_int = 100
sleep_time = 10
# safety settings for Gemini pro
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

def prompt(model, type_llm, prompt):
    if type_llm == 'gemini_pro':
        config = {'candidate_count':1, "max_output_tokens": 2048, "temperature": 0, "top_p": 0.99, "top_k": 32}
        responses = model.generate_content(
            prompt,
            generation_config=config,
            safety_settings = safety_settings
        )
        return responses.text
    elif type_llm =='chatGPT':
        response = client.chat.completions.create(
            model="GPT-35-turbo-WebQSP-Vanilla", # model = "deployment_name"
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        return response.choices[0].message.content


### FewShot
def fewshot(model,city, type_method,type_llm,  data_user_test, user_kws_train,  map_rest_id2int, new_results_res_kw, num_kws_user, num_kws_rest, samples, label_samples, gpt_prefix='Output:'):
    if type_llm == "chatGPT":
        if type_method == '1_shot':
            temp_fewshot = '''
            You are a restaurant recommendation system. You recommend restaurant for new user who only provide a few keywords to indicate preference. 
            You are given a list of keywords provided by new user, a list of candidate restaurants (Format: [restaurant_1, restaurant_2,...]), and a list of keywords describing each restaurant in the following format restaurant_1 (keyword 1, keyword 2,...). You need to re-rank the restaurant candidates such that the restaurants ranked higher are the ones the user most likely wants to go. 
            Below are the examples for your reference. 
            Input keywords: {}. 
            Restaurant list: {}. 
            Restaurant-keyword: {}. 
            Output: {} 

            Based on the example above, please perform the following task:
            These are keywords describing my preference: {}.
            The candidate restaurant set for me is enclosed in square brackets, with the restaurants separated by commas (Format: [restaurant_1, restaurant_2,...]) is: {}
            Keywords associated with candidate restaurants have the following form: restaurant_1 (keyword 1, keyword 2,...) are {}.
            Input: Please suggest the 15 most suitable restaurants for me from the candidate set that I will visit them, according to the user and candidate restaurant keywords I provided above.
            Output: Must include 15 restaurants in the candidate set. No explanation. Desired format is string: restaurant_1, restaurant_2, ... 
            '''
        elif type_method == '2_shots':
            temp_fewshot = '''
            You are a restaurant recommendation system. You recommend restaurant for new user who only provide a few keywords to indicate preference. 
            You are given a list of keywords provided by new user, a list of candidate restaurants (Format: [restaurant_1, restaurant_2,...]), and a list of keywords describing each restaurant in the following format restaurant_1 (keyword 1, keyword 2,...). You need to re-rank the restaurant candidates such that the restaurants ranked higher are the ones the user most likely wants to go. 
            Below are the examples for your reference. 
            Input keywords: {}. 
            Restaurant list: {}. 
            Restaurant-keyword: {}. 
            Output: {} 

            Input keywords: {}. 
            Restaurant list: {}. 
            Restaurant-keyword: {}. 
            Output: {} 

            Based on the examples above, please perform the following task:
            These are keywords describing my preference: {}.
            The candidate restaurant set for me is enclosed in square brackets, with the restaurants separated by commas (Format: [restaurant_1, restaurant_2,...]) is: {}
            Keywords associated with candidate restaurants have the following form: restaurant_1 (keyword 1, keyword 2,...) are {}.
            Input: Please suggest the 15 most suitable restaurants for me from the candidate set that I will visit them, according to the user and candidate restaurant keywords I provided above.
            Output: Must include 15 restaurants in the candidate set. No explanation. Desired format is string: restaurant_1, restaurant_2, ... 
            '''
        elif type_method == '3_shots':
            temp_fewshot = '''
            You are a restaurant recommendation system. You recommend restaurant for new user who only provide a few keywords to indicate preference. 
            You are given a list of keywords provided by new user, a list of candidate restaurants (Format: [restaurant_1, restaurant_2,...]), and a list of keywords describing each restaurant in the following format restaurant_1 (keyword 1, keyword 2,...). You need to re-rank the restaurant candidates such that the restaurants ranked higher are the ones the user most likely wants to go. 
            Below are the examples for your reference. 
            Input keywords: {}. 
            Restaurant list: {}. 
            Restaurant-keyword: {}. 
            Output: {} 

            Input keywords: {}. 
            Restaurant list: {}. 
            Restaurant-keyword: {}. 
            Output: {} 

            Input keywords: {}. 
            Restaurant list: {}. 
            Restaurant-keyword: {}. 
            Output: {} 

            Based on the examples above, please perform the following task:
            These are keywords describing my preference: {}.
            The candidate restaurant set for me is enclosed in square brackets, with the restaurants separated by commas (Format: [restaurant_1, restaurant_2,...]) is: {}
            Keywords associated with candidate restaurants have the following form: restaurant_1 (keyword 1, keyword 2,...) are {}.
            Input: Please suggest the 15 most suitable restaurants for me from the candidate set that I will visit them, according to the user and candidate restaurant keywords I provided above.
            Output: Must include 15 restaurants in the candidate set. No explanation. Desired format is string: restaurant_1, restaurant_2, ... 
            '''
    elif type_llm == 'gemini_pro':
        if type_method == '1_shot':
            temp_fewshot = '''
            Assume you are a restaurant recommendation system. For example: 
            There are the keywords that user often mention when wanting to choose restaurants: {}.
            The candidate restaurant set for user is enclosed in square brackets, with the restaurants separated by commas (Format: [restaurant_1, restaurant_2,...]) is: {}
            Keywords associated with candidate restaurants have the following form: restaurant_1 (keyword 1, keyword 2,...) are {}.
            You should rank a list of recommendations for user as follows: {}

            Based on the example above, where those users exhibit similar behavior to mine, please perform the following task:
            There are the keywords that I often mention when wanting to choose restaurants: {}.
            The candidate restaurant set for me is enclosed in square brackets, with the restaurants separated by commas (Format: [restaurant_1, restaurant_2,...]) is: {}
            Keywords associated with candidate restaurants have the following form: restaurant_1 (keyword 1, keyword 2,...) are {}.
            Input: Please suggest the 15 most suitable restaurants for me from the candidate set that I will visit them, according to the user and candidate restaurant keywords I provided above.
            Output: Must include 15 restaurants in the candidate set. No explanation. Desired format is string: restaurant_1, restaurant_2, ... 
            '''
        elif type_method == '2_shots':
            temp_fewshot = '''
            Assume you are a restaurant recommendation system. For example: 
            Example 1: There are the keywords that user often mention when wanting to choose restaurants: {}.
            The candidate restaurant set for user is enclosed in square brackets, with the restaurants separated by commas (Format: [restaurant_1, restaurant_2,...]) is: {}
            Keywords associated with candidate restaurants have the following form: restaurant_1 (keyword 1, keyword 2,...) are {}.
            You should rank a list of recommendations for user as follows: {}
            Example 2: There are the keywords that user often mention when wanting to choose restaurants: {}.
            The candidate restaurant set for user is enclosed in square brackets, with the restaurants separated by commas (Format: [restaurant_1, restaurant_2,...]) is: {}
            Keywords associated with candidate restaurants have the following form: restaurant_1 (keyword 1, keyword 2,...) are {}.
            You should rank a list of recommendations for user as follows: {}

            Based on 2 examples above, please perform the following task:
            There are the keywords that I often mention when wanting to choose restaurants: {}.
            The candidate restaurant set for me is enclosed in square brackets, with the restaurants separated by commas (Format: [restaurant_1, restaurant_2,...]) is: {}
            Keywords associated with candidate restaurants have the following form: restaurant_1 (keyword 1, keyword 2,...) are {}.
            Input: Please suggest the 15 most suitable restaurants for me from the candidate set that I will visit them, according to the user and candidate restaurant keywords I provided above.
            Output: Must include 15 restaurants in the candidate set. No explanation. Desired format is string: restaurant_1, restaurant_2, ... 
            '''
        elif type_method == '3_shots':
            temp_fewshot = '''
            Assume you are a restaurant recommendation system. For example: 
            Example 1: There are the keywords that user often mention when wanting to choose restaurants: {}.
            The candidate restaurant set for user is enclosed in square brackets, with the restaurants separated by commas (Format: [restaurant_1, restaurant_2,...]) is: {}
            Keywords associated with candidate restaurants have the following form: restaurant_1 (keyword 1, keyword 2,...) are {}.
            You should rank a list of recommendations for user as follows: {}
            Example 2: There are the keywords that user often mention when wanting to choose restaurants: {}.
            The candidate restaurant set for user is enclosed in square brackets, with the restaurants separated by commas (Format: [restaurant_1, restaurant_2,...]) is: {}
            Keywords associated with candidate restaurants have the following form: restaurant_1 (keyword 1, keyword 2,...) are {}.
            You should rank a list of recommendations for user as follows: {}
            Example 3: There are the keywords that user often mention when wanting to choose restaurants: {}.
            The candidate restaurant set for user is enclosed in square brackets, with the restaurants separated by commas (Format: [restaurant_1, restaurant_2,...]) is: {}
            Keywords associated with candidate restaurants have the following form: restaurant_1 (keyword 1, keyword 2,...) are {}.
            You should rank a list of recommendations for user as follows: {}
            Based on 3 examples above, please perform the following task:
            There are the keywords that I often mention when wanting to choose restaurants: {}.
            The candidate restaurant set for me is enclosed in square brackets, with the restaurants separated by commas (Format: [restaurant_1, restaurant_2,...]) is: {}
            Keywords associated with candidate restaurants have the following form: restaurant_1 (keyword 1, keyword 2,...) are {}.
            Input: Please suggest the 15 most suitable restaurants for me from the candidate set that I will visit them, according to the user and candidate restaurant keywords I provided above.
            Output: Must include 15 restaurants in the candidate set. No explanation. Desired format is string: restaurant_1, restaurant_2, ... 
            '''
    user_rank = dict()
    i = 0
    len_rank = None
    folder_path_result_rerank = f'{root_dir}results_rerank/{city}'
    if not os.path.exists(folder_path_result_rerank):
        os.makedirs(folder_path_result_rerank)
    file_path_ = folder_path_result_rerank+ f"/{type_method}_{num_kws_user}_{num_kws_rest}.json"
    if os.path.isfile(file_path_):
        print("File exists")
        with open(file_path_, "r") as file:
            user_rank = json.load(file)
        len_rank = len(user_rank.keys())
    for uid in tqdm(data_user_test.keys(), total=len(data_user_test)):
        if len_rank is not None:
            if i <= len_rank:
                continue
        # print(i)
        # print(uid)
        if (i+1) % sleep_int == 0:
            time.sleep(sleep_time)
        user_kw = data_user_test[uid]['kw'][: num_kws_user]  # use 5 kws for user
        res_candidate = list(map(int,[str(map_rest_id2int[cand]) for cand in data_user_test[uid]['candidate']]))
        ### selecting randomly users for examples
        user_train = random.choice(list(samples.keys()))
        user_train_2 = random.choice(list(samples.keys()))
        user_train_3 = random.choice(list(samples.keys()))

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
            input = temp_fewshot.format(', '.join(user_train_kw), res_candidate_train , cand_kw_fn_fewshot(user_train, new_results_res_kw,samples, map_rest_id2int, 20, num_kws_rest),label_res,
                                     ', '.join(user_kw),res_candidate, cand_kw_fn(uid, new_results_res_kw,data_user_test, map_rest_id2int, 20, num_kws_rest))
        elif type_method == '2_shots':
            input = temp_fewshot.format(', '.join(user_train_kw), res_candidate_train , cand_kw_fn_fewshot(user_train, new_results_res_kw,samples, map_rest_id2int, 20, num_kws_rest),label_res,
                                        ', '.join(user_train_kw_2), res_candidate_train_2 , cand_kw_fn_fewshot(user_train_2, new_results_res_kw,samples, map_rest_id2int, 20, num_kws_rest),label_res_2,
                                     ', '.join(user_kw),res_candidate, cand_kw_fn(uid, new_results_res_kw,data_user_test, map_rest_id2int, 20, num_kws_rest))
        elif type_method == '3_shots':
            input = temp_fewshot.format(', '.join(user_train_kw), res_candidate_train , cand_kw_fn_fewshot(user_train, new_results_res_kw,samples, map_rest_id2int, 20, num_kws_rest),label_res,
                                        ', '.join(user_train_kw_2), res_candidate_train_2 , cand_kw_fn_fewshot(user_train_2, new_results_res_kw,samples, map_rest_id2int, 20, num_kws_rest),label_res_2,
                                        ', '.join(user_train_kw_3), res_candidate_train_3 , cand_kw_fn_fewshot(user_train_3, new_results_res_kw,samples, map_rest_id2int, 20, num_kws_rest),label_res_3,
                                     ', '.join(user_kw),res_candidate, cand_kw_fn(uid, new_results_res_kw,data_user_test, map_rest_id2int, 20, num_kws_rest))

        predictions = prompt(model,type_llm, input)
        i= i+1        
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
def zeroshot(model, type_llm, data_user_test,  map_rest_id2int, new_results_res_kw, num_kws_user, num_kws_rest, gpt_prefix='Output:'):
    if city == 'tripAdvisor':
        # temp = '''
        # Assume you are a hotel recommendation system.
        # The keywords I frequently use when selecting hotels are: {}.
        # The set of candidate hotel for me, listed within square brackets and separated by commas (Format: [hotel_id_1, hotel_id_2, ...]), is: {}.
        # Keywords associated with the candidate hotels are in the following format: hotel_id_1 (keyword 1, keyword 2, ...) are: {}.
        # Input: Based on the provided user and candidate hotel keywords, please recommend the 15 most suitable hotels for me from the candidate set that I will visit.
        # Output: The output must include 15 hotels from the candidate hotel set, formatted as a string: hotel_id_1, hotel_id_2, ...
        # '''
        temp = '''
        There are the keywords that I often mention when wanting to choose hotels: {}.
        The candidate hotel set for me is enclosed in square brackets, with the hotels separated by commas (Format: [hotel_id_1, hotel_id_2,...]) is: {}
        Keywords associated with candidate hotels are in the following format: hotel_id_1 (keyword 1, keyword 2,...) are {}.
        Input: Please suggest the 15 most suitable hotels for me from the candidate set that I will visit them, according to the user and candidate hotel keywords I provided above.
        Output: Must include 15 hotels in the candidate hotel set. No explanation. Desired format is string: hotel_id_1, hotel_id_2, ... 
        '''
    else:
        temp = '''
        There are the keywords that I often mention when wanting to choose restaurants: {}.
        The candidate restaurant set for me is enclosed in square brackets, with the restaurants separated by commas (Format: [restaurant_id_1, restaurant_id_2,...]) is: {}
        Keywords associated with candidate restaurants have the following form: restaurant_id_1 (keyword 1, keyword 2,...) are {}.
        Input: Please suggest the 15 most suitable restaurants for me from the candidate set that I will visit them, according to the user and candidate restaurant keywords I provided above.
        Output: Must include 15 restaurants in the candidate restaurant set. No explanation. Desired format is string: restaurant_id_1, restaurant_id_2, ... 
        '''

    user_rank = dict()
    # user_shuffle = dict()
    i = 0
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
    for uid in tqdm(data_user_test.keys(), total=len(data_user_test)):
        i = i+1
        if i %1000 == 0:
            time.sleep(120)
        if len_rank is not None:
            if i <= len_rank:
                continue
        user_kw = data_user_test[uid]['kw'][:num_kws_user] 
        res_candidate = list(map(int,[str(map_rest_id2int[cand]) for cand in data_user_test[uid]['candidate']])) 
        # random.shuffle(res_candidate)
        # user_shuffle[uid] = res_candidate
        input = temp.format(', '.join(user_kw),res_candidate, cand_kw_fn(uid, new_results_res_kw, data_user_test, map_rest_id2int, 20, num_kws_rest))
        flag = False
        while flag is False:
            try:
                predictions = prompt(model, type_llm, input)
                # print(predictions)
                flag = True
                i= i+1
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

        # pred_ = predictions.split(',')
        # pred = []
        # for aa in pred_:
        #     pred.append(extract_numbers(aa))
        user_rank[uid] = list(map(int, pred))
        with open(file_path_, "w") as json_file:
            json.dump(user_rank, json_file)
        # with open(file_shuffle, "w") as json_file:
        #     json.dump(user_shuffle, json_file)

    return user_rank

### eval
def eval(user_rank, is_base = False):
    prec_final_2, rec_final_2, f1_final_2 = [],[],[]
    result_str = ''
    if is_base is True:
        eval = [1,3,5, 10, 15,20]
    else:
        eval = [1,3,5,10,15]
    for k in eval:
        prec_final, rec_final, f1_final = [],[],[]
        for uid_ in user_rank.keys():
            pred =  [int(pred_) for pred_ in user_rank[uid_]]
            prec, rec, f1= quick_eval(pred[:k], u2rs[uid_])
            prec_final.append(prec)
            rec_final.append(rec) 
            f1_final.append(f1) 
        prec_final_2.append(np.mean(prec_final))
        rec_final_2.append(np.mean(rec_final))
        print(f'Precision@{k}: {np.mean(prec_final)}, recall@{k}: {np.mean(rec_final)}, f1@{k}: {np.mean(f1_final)}')
        result_str += f'Precision@{k}: {np.mean(prec_final)}, recall@{k}: {np.mean(rec_final)}, f1@{k}: {np.mean(f1_final)} \n'
    print(len(user_rank.keys()))
    for i in range(len(eval)):
        print(f'{prec_final_2[i]} {rec_final_2[i]}', end=' ')
        result_str += f'{prec_final_2[i]} {rec_final_2[i]} '
    return result_str, prec_final_2, rec_final_2


if __name__ == '__main__':
    listcity = ['charlotte', 'edinburgh', 'lasvegas', 'london', 'phoenix', 'pittsburgh', 'singapore']
    parser = argparse.ArgumentParser('LLM re-ranking RecSys')
    parser.add_argument('--type_method', type=str, default= 'zeroshot', help='zeroshot,1_shot, 2_shots, 3_shots')
    parser.add_argument('--num_kws_user', type=int, default= 3)
    parser.add_argument('--num_kws_rest', type=int, default= 5)
    parser.add_argument('--city', type=str, default='tripAdvisor', help=f'choose city{listcity}')
    parser.add_argument('--type_LLM', type=str, default='gemini_pro', help='gemini_pro, chatGPT,...')
    parser.add_argument('--api_key', type=str, default=None, help='API key')


    args = parser.parse_args()
    root_dir = 'reRanker/'

    run_list_kws_for_user = [5]
    run_list_kws_for_rest = [10]
    list_method = ['zeroshot']
    # print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    ### Load data
    city = args.city
    data_user_test_ = read_json(f"data/{args.city}/{args.city}_knn2rest.json")
    rest_kws = read_json(f"data/score/{args.city}-keywords-TFIUF.json")
    user_kws_train_ = read_json(f'data/{args.city}/{args.city}_user2candidate.json') #user-train


    ## review filesS
    if city == 'tripAdvisor':
        is_tripAdvisor = True
    else: 
        is_tripAdvisor= False
    gt_file = 'data/reviews/{}.csv'.format(args.city)
    gt, u2rs, map_rest_id2int_ = prepare_user2rests(gt_file, is_tripAdvisor = is_tripAdvisor)


    new_results_res_kw_ = get_kw_for_rest(rest_kws, map_rest_id2int_)

    # sample for fewshot
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

    print('baseline')
    result_str = ''
    user_cand = dict()
    i = 0
    prec_final_2, rec_final_2, f1_final_2 = [],[],[]
    eval_ = [1,3,5,10,15,20]
    for k in eval_:
        prec_final, rec_final, f1_final = [],[],[]
        for uid in data_user_test_.keys():
            # print(uid)
            quick_preds = [map_rest_id2int_[can] for can in data_user_test_[uid]['candidate']]
            prec, rec, f1= quick_eval(quick_preds[:k], u2rs[uid])
            prec_final.append(prec)
            rec_final.append(rec)
            f1_final.append(f1)
            i = i+1
        prec_final_2.append(np.mean(prec_final))
        rec_final_2.append(np.mean(rec_final))
        print(f'precision@{k}: {np.mean(prec_final)}, recall@{k}: {np.mean(rec_final)}, f1@{k}: {np.mean(f1_final)}')
        result_str += f'Precision@{k}: {np.mean(prec_final)}, recall@{k}: {np.mean(rec_final)}, f1@{k}: {np.mean(f1_final)} \n'
    for i in range(len(eval_)):
        print(f'{prec_final_2[i]} {rec_final_2[i]}', end=' ')
        result_str += f'{prec_final_2[i]} {rec_final_2[i]} '
    print(len(data_user_test_.keys()))
    with open(file_path, 'w') as file:
        file.write('Baseline \n')
        file.write(result_str)
        file.write("\n")

    print('Begin run LLM tests')
    result_json = dict()
    for method_ in list_method:
        result_json[method_] = dict()
        for kws_user in run_list_kws_for_user:
            args.num_kws_user = kws_user
            result_json[method_][kws_user] = dict()
            for kws_rest in run_list_kws_for_rest:
                result_json[method_][kws_user][kws_rest] = dict()
                args.num_kws_rest = kws_rest
                args.type_method = method_
                print('\nargs: ', args)
                data_args = vars(args)
                with open(file_path, 'a') as file:
                # Write log
                    file.write("\n")
                    json.dump(data_args, file)
                    file.write("\n")

                if args.type_LLM == 'gemini_pro':
                    os.getenv('GOOGLE_API_KEY')
                    genai.configure(api_key=args.api_key)
                    model = genai.GenerativeModel("gemini-pro")
                    if args.type_method == '1_shot' or args.type_method =='2_shots' or args.type_method == '3_shots':
                        user_rank = fewshot(model,city, args.type_method,  args.type_LLM, data_user_test_, user_kws_train_,  map_rest_id2int_, new_results_res_kw_, args.num_kws_user, args.num_kws_rest, samples, label_samples)
                    elif args.type_method == 'zeroshot':
                        user_rank = zeroshot(model, args.type_LLM, data_user_test_, map_rest_id2int_, new_results_res_kw_, args.num_kws_user, args.num_kws_rest)                                        
                elif args.type_LLM == 'chatGPT':
                    # define model
                    client = AzureOpenAI(
                        azure_endpoint = "https://webqsp.openai.azure.com/", 
                        api_key=os.getenv("AZURE_OPENAI_KEY"),  
                        api_version="2024-02-15-preview"
                        )
                    if args.type_method == '1_shot' or args.type_method =='2_shots' or args.type_method == '3_shots':
                        user_rank = fewshot(client, city, args.type_method, args.type_LLM, data_user_test_, user_kws_train_,  map_rest_id2int_, new_results_res_kw_, args.num_kws_user, args.num_kws_rest, samples, label_samples)
                    elif args.type_method == 'zeroshot':
                        user_rank = zeroshot(client, args.type_LLM, data_user_test_, map_rest_id2int_, new_results_res_kw_, args.num_kws_user, args.num_kws_rest)
                result_str, prec_, recall_ = eval(user_rank)
                result_json[method_][kws_user][kws_rest]['prec'] = prec_
                result_json[method_][kws_user][kws_rest]['recall'] = recall_

                with open(file_path_json, "a") as json_file:
                    json.dump(result_json, json_file)
                with open(file_path, 'a') as file:
                    file.write(result_str)
                    file.write("\n")
                print('args: ', args)
                print('slepping ...')