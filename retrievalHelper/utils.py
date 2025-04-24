import random
import torch
import numpy as np
import json
from tqdm import tqdm
import pandas as pd

def setSeed(random_seed):
    torch.manual_seed(random_seed) # cpu
    torch.cuda.manual_seed(random_seed) #gpu
    np.random.seed(random_seed) #numpy
    random.seed(random_seed) #random and transforms
    torch.backends.cudnn.deterministic=True # cudnn

def data_reviewLoader(city):
    '''
    input: 
        city: name city
    output:
        train, test data
    '''
    filename = ['_train', '_test']
    datas = []
    for fn in filename:    
        f = open(f'./data/keywords/{city}-keywords{fn}.json')
        data = json.load(f)
        keys = [ ii for ii in data]
        print("info from extracted file")
        print(f"Number of keyword: {len(data[keys[0]])}")
        datas.append(data)
    return datas


def extract_users(info):
    '''
    input:
        info: data['np2users']
    output:
        list_user, user2kw
    '''
    l_user, user2kw = [], []
    for ii in info:
        lus = info[ii]
        for u in lus:
            if u not in l_user:
                l_user.append(u)
                user2kw.append([])
            idx = l_user.index(u)
            user2kw[idx].append(ii)
    return l_user, user2kw

def load_kwScore(city, edgeType):
    """
    for raw 
    keywordscore = {user: {kw: freq}}
    for IUF
    keywordscore = {rest: {kw: freq}}
    """
    f = open(f'./data/score/{city}-keywords-TF{edgeType}.json')
    keywordScore = json.load(f)
    f = open(f'./data/score/{city}-keywords-frequency.json')
    keywordFrequence = json.load(f)
    return keywordScore, keywordFrequence    

def load_groundTruth(gt_file):
    gt = pd.read_csv(gt_file)
    print("number of review: ", len(gt))
    u2rs = {}
    for uid, rid, r in zip(gt['user_id'], gt['rest_id'], gt['rating']):
        if uid not in u2rs:
            u2rs[uid] = []
        u2rs[uid].append((str(rid), r))
    return u2rs

def getRestLB(data):
    '''
    data = train_data = ['np2rest':{},...]
    --> return list of restaurant
    '''
    rest_Label = []
    for idx, kw in enumerate(data):
        listkw_rest = data[kw]
        for rest in listkw_rest:
            if rest not in rest_Label:
              rest_Label.append(rest)
    return rest_Label

def label_ftColab(train_users, test_users, gt, no_rest, rest_Label):
    label_train = gt2label(train_users, gt, no_rest, "label_train.npy", rest_Label)
    label_test = gt2label(test_users, gt, no_rest, "label_test.npy", rest_Label)

    return label_train, label_test

def gt2label(list_user, gt, no_rest, filename, rest_Label):
    LB = []
    for user in tqdm(list_user):
        visitedRest = gt[user]
        newlabel = np.zeros(no_rest)
        for rest, rate in visitedRest:
            try:
                idx = rest_Label.index(rest)
                newlabel[idx] = 1
            except Exception as e:
                pass
        LB.append(newlabel)
    LB = np.asarray(LB)
    return LB

def userKW2FT(userKw, kwEB_pad, kw_data, sameShape = True):
    usersFT = []
    for idx, kws in enumerate(userKw):
        topK = len(kws)
        userFT = []
        for kw in kws:
            kwIdx = kw_data.index(kw)
            em = np.asarray(kwEB_pad[kwIdx])
            userFT.append(em)
        userFT = np.vstack(userFT)
        usersFT.append(userFT)
    if sameShape:
        return np.asarray(usersFT)
    else:
        return usersFT

def mean(x):
    return np.mean(np.asarray(x))
    
def extractResult(lResults):
    p = [x[0] for x in lResults]
    r = [x[1] for x in lResults]
    f = [x[2] for x in lResults]
    return p, r, f


def procesTest(test_users, test_users2kw, idx, KNN, restGraph, returnTop = False):
    testUser = test_users[idx]
    testkey = test_users2kw[idx]
    testkey = KNN.get_topK_Key(testkey)

    topK_Key, keyfrequency = restGraph.retrievalKey(testkey)
    if returnTop:
        topUser = restGraph.key_score2simUser(topK_Key, keyfrequency)
        return testUser, topK_Key, keyfrequency, topUser

    return testUser, topK_Key , "", ""   



class regionHelper(object):
    """docstring for regionHelper"""
    def __init__(self, l_rest):
        super(regionHelper, self).__init__()
        df2 = pd.read_csv('./data/reviews/hotel.csv')
        hotel_region_dict = dict(zip(df2['hotelID'], df2['region_id']))
        rest2city = []
        for rest in l_rest:
            rest = int(rest)
            rest2city.append(hotel_region_dict[rest])
        self.rest2city = np.asarray(rest2city)
        with open('./data/reviews/user2region.json', 'r') as f:
            self.author_region = json.load(f)


    def query(self, id):
        tmp_reg = self.author_region[id]
        marker = np.zeros(len(self.rest2city))
        for reg in tmp_reg:
            tmp = np.where(self.rest2city == reg)[0]
            marker[tmp] = 1
        return marker
        