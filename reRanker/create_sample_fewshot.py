import json
import pandas as pd
import os

# Open the JSON file for reading
with open('data/reviews/splits.json', 'r') as file:
    # Load the JSON data from the file
    data = json.load(file)

city = 'london'

user_train = data[city]['train']

rv_ = pd.read_csv(f'/home/ubuntu/duc.nm195858/KALM4Rec/data/reviews/{city}.csv')

total_res = set(rv_['rest_id'].unique().tolist())

import random
random.seed(1234)
uid_train_rest = dict()
label_ = dict()
count_10 = 0
for uid in user_train:
    hist_res = rv_[rv_['user_id'] == uid].sort_values('rating', ascending=False)['rest_id'].values.tolist()
    rest_res = list(total_res - set(hist_res))
    if len(hist_res) <= 5:
        fill_res_len = 20-len(hist_res)
        random_fill = random.sample(rest_res, fill_res_len)
        new_res_candi = hist_res + random_fill
        label_res = hist_res + random_fill
    else:
        random_fill = random.sample(rest_res, 15)
        label_res = hist_res[:5] + random_fill
        new_res_candi = hist_res[:5] + random_fill
        count_10 +=1
    random.shuffle(new_res_candi)
    uid_train_rest[uid] =  new_res_candi
    label_[uid] = label_res
print(count_10)
folder_path = 'data/fewshot_samples'
if not os.path.exists(folder_path):
        # If it doesn't exist, create it
        os.makedirs(folder_path)

file_path = f'data/fewshot_samples/{city}_5.json'
with open(file_path, 'w') as file:
    # Save the dictionary as JSON
    json.dump(uid_train_rest, file)

file_path = f'data/fewshot_samples/{city}_label_5.json'
with open(file_path, 'w') as file:
    # Save the dictionary as JSON
    json.dump(label_, file)

