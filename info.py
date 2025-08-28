import pandas as pd
import json
from pprint import pprint
import json 
import os 
from tqdm import tqdm 
import time 
import numpy as np
from reRanker.utils import *
from retrievalHelper.utils import *
import random
import argparse
import time
import ast
import re


listcity = ['edinburgh', 'london', 'singapore']
parser = argparse.ArgumentParser('info result Kalm4Rec')
parser.add_argument('--city', type=str, default='singapore', help=f'choose city{listcity}')
parser.add_argument('--baseline', type=bool, default=False, help='print baseline')
parser.add_argument('--fileName', type=str, help='filename')


args = parser.parse_args()
root_dir = 'reRanker/'

### Load data
city = args.city
data_user_test = read_json(f"data/out2LLMs/{args.city}_knn2rest.json")
# rest_kws = read_json(f"data/score/{args.city}-keywords-TFIUF.json")
# user_kws_train = read_json(f'data/out2LLMs/{args.city}_user2candidate.json') #user-train


## review filesS
if city == 'tripAdvisor':
    is_tripAdvisor = True
else: 
    is_tripAdvisor= False
gt_file = 'data/reviews/{}.csv'.format(args.city)
gt, u2rs, map_rest_id2int = prepare_user2rests(gt_file, is_tripAdvisor = is_tripAdvisor)

if args.baseline:
    print('baseline')
    print(city)
    user_dict = {}
    for uid in data_user_test.keys():
        user_dict[uid] = [map_rest_id2int[can] for can in data_user_test[uid]['candidate']]
    evalAll(user_dict, u2rs)

else:
    pass
