import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import string
import shutil
import os
from sentence_transformers import SentenceTransformer, util
import argparse
from retrievalHelper.utils import *
from retrievalHelper.u4Res import *
from retrievalHelper.u4KNN import *
from reRanker.utils import read_json


def get_kw_for_rest(rest_kws, map_rest_id2int ):
    res2kw_all = dict()
    for res, kw_score in rest_kws.items():
        res2kw_all[res] = []
        for a in kw_score:
            res2kw_all[res].append(a[0])

    new_results_res_kw = {}
    counter = 0
    for res, kws in res2kw_all.items():
        if res in map_rest_id2int:
            new_results_res_kw[map_rest_id2int[res]] = kws
        else:
            counter += 1
    print(f"Num of missing: {counter}")
    return new_results_res_kw

def getFilter(listCheck, resMatrix, userMatrix, interact = True, value = 0.4, freq = 4):
    filterSimi = []
    filterNotSimi = []
    for test in listCheck:
        testembds = np.asarray(resMatrix[test][:kws_for_rest])
        resultMat = userMatrix @ testembds.T
        count = np.sum(resultMat > value)
        if count >= freq:
            filterSimi.append(test)
        else:
            filterNotSimi.append(test)
    if interact:
        return filterSimi
    return filterNotSimi


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='london', help=f'choose city')
    parser.add_argument('--kws_for_user', type=int, default=3, help='kws_for_user')
    parser.add_argument('--kws_for_rest', type=int, default=5, help='kws_for_rest')
    parser.add_argument('--type', type=str, default='mct', help=f'mct: multiple choice + token, mcl: multiple choice + list, list')


    args = parser.parse_args()

    print("args:", args)
    city = args.city
    trainDat, testDat = data_reviewLoader(city)
    train_users, train_users2kw = extract_users(trainDat['np2users'])
    test_users, test_users2kw = extract_users(testDat['np2users'])
    gt = load_groundTruth(f'./data/reviews/{city}.csv')
    keywordScore, keywordFrequence = load_kwScore(city, "IUF")
    restGraph = retaurantReviewG([trainDat, keywordScore, keywordFrequence, 20,  "IUF", gt])
    KNN = neighbor4kw(f'{city}_kwSenEB_pad', testDat,  restGraph)
    rest_Label = getRestLB(trainDat['np2rests'])
    restID2int = {x: rest_Label.index(x) for x in rest_Label}
    print(f"number of user: {len(train_users)}, number of item: {len(rest_Label)}")
    label_train, label_test = label_ftColab(train_users, test_users, gt, restGraph.numRest, rest_Label)
    trainLB = np.asarray(label_train)
    testLB = np.asarray(label_test)

    rest_kws = read_json(f"./data/score/{city}-keywords-TFIUF.json")
    train_res_kw = get_kw_for_rest(rest_kws, restID2int)
    keywordScore4User,_ = load_kwScore(city, "IRF")

    alphabet = string.ascii_uppercase
    letters = alphabet[:20]

    with open(f"./data/out2LLMs/retrievalSample_{city}.json", "r") as f:
        loaded_data = json.load(f)

    listPosNeg = []
    for uid in tqdm(range(len(trainLB))):
        listInteract = trainLB[uid]
        interacted = set(np.where(listInteract == 1)[0])   # convert to set for faster lookup
        selected = loaded_data[uid]
        pos = [x for x in selected if x in interacted]
        neg = [x for x in selected if x not in interacted]
        listPosNeg.append((uid, selected, pos, neg))


    random.shuffle(listPosNeg)
    len(listPosNeg)   
    listPosNeg = listPosNeg[:2000] 

    listData = []
    counter = 0
    shuffle = 20
    hardsample = 40

    # keep huge part is not being change to learn the pattern
    for uid, interacted, pos_items, sampled_neg_items in tqdm(listPosNeg):
        if len(sampled_neg_items) + len(pos_items) != 20:
            continue
        if len(sampled_neg_items) == 20:
            continue
        if len(pos_items) == 20:
            continue
        lus = keywordScore4User[train_users[uid]][:args.kws_for_user]
        lu = [x for x,y in lus]
        negs = [x for x in sampled_neg_items]
        poss = [x for x in pos_items]
        
        
        tmp = random.randint(1,100)
        if tmp < shuffle:
            random.shuffle(negs)
            random.shuffle(poss)
        candidate = []
        negcount = 0
        poscount = 0
        for x in interacted:
            if x in poss:
                candidate.append(poss[poscount])
                poscount += 1
            else:
                candidate.append(negs[negcount])
                negcount += 1
        if (tmp < hardsample) and (tmp > shuffle):
            randPosition = random.randint(0, len(negs)-1)
            negPos = candidate.index(negs[randPosition])
            posPos = candidate.index(poss[0])
            candidate[posPos] = candidate[negPos]
            candidate[negPos] = poss[0]
        
        outLb = [candidate.index(x) for x in poss]
        lc = [train_res_kw[x][:args.kws_for_rest] for x in candidate]
        listData.append((lu, lc,outLb))


    # mct
    counterAppear = [0] * len(letters)
    listTrain = []
    for datapoint in listData[:4000]:
        npos = len(datapoint[-1])

        # get all positive item, move to first
        for ii in range(min(npos, 10)):
            tmp = ord(letters[datapoint[2][ii]]) - ord('A')
            counterAppear[tmp] += 1
            input1 = ', '.join(datapoint[0])
            allkw = datapoint[1]
            input2 = ' '.join(map(lambda item: f'{letters[item[0]]}. ({", ".join(item[1])}) ;\n', enumerate(allkw)))
            lt = [letters[x] for x in datapoint[2]]
            listTrain.append({'user': input1, 'input': input2, 'output': lt, 'top': letters[datapoint[2][ii]]})

    for l,c in zip(counterAppear, letters):
        print(l, c)

    len(listTrain)
    with open(f'./data/out2LLMs/train_data_{city}.json', 'w', encoding='utf-8') as f:
        json.dump(listTrain, f, ensure_ascii=False, indent=2)    
