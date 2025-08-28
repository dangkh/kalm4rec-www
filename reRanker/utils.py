import json
import pandas as pd
import random
import numpy as np
from sklearn.metrics import ndcg_score

def read_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

### Map restaurants to ids and obtain the set of restaurants the user has visited
def prepare_user2rests(gt_file, is_tripAdvisor):
    gt = pd.read_csv(gt_file)
    u2rs = {}
    i = 0
    map_rest_id2int = dict()
    for uid, rid, r in zip(gt['user_id'], gt['rest_id'], gt['rating']):
        if is_tripAdvisor:
            rid_str = str(rid)
            if rid_str not in map_rest_id2int:
                map_rest_id2int[rid_str] = i
                i +=1
            if uid not in u2rs:
                u2rs[uid] = []
            u2rs[uid].append((map_rest_id2int[rid_str], r))
        else:
            if rid not in map_rest_id2int:
                map_rest_id2int[rid] = i
                i +=1
            if uid not in u2rs:
                u2rs[uid] = []
            u2rs[uid].append((map_rest_id2int[rid], r))
    return gt,u2rs,map_rest_id2int

### retrieve the keywords for candidate restaurant and return them as a string
def cand_kw_fn(uid_, result_dict,data,map_rest_id2int, topCandidates = 20, topKWs = 20):
    cand_kw = {}
    for cand in data[uid_]['candidate'][: topCandidates]:
            if map_rest_id2int[cand] in result_dict:
                if len(result_dict[map_rest_id2int[cand]]) > topKWs:
                    cand_kw[map_rest_id2int[cand]] = result_dict[map_rest_id2int[cand]][:topKWs]
                else:
                    cand_kw[map_rest_id2int[cand]] = result_dict[map_rest_id2int[cand]]
            else:
                cand_kw[map_rest_id2int[cand]] = []
    result_string = ', '.join(f'{key} ({", ".join(value)})' for key, value in cand_kw.items())
    return result_string

### retrieve the keywords for candidate restaurant and return them as a string, in fewshot cases.
def cand_kw_fn_fewshot(uid_, result_dict, data_, map_rest_id2int, topCandidates = 20, topKWs = 20):
    cand_kw = {}
    for cand in data_[uid_][: topCandidates]:
            if map_rest_id2int[cand] in result_dict:
                if len(result_dict[map_rest_id2int[cand]]) > topKWs:
                    cand_kw[map_rest_id2int[cand]] = result_dict[map_rest_id2int[cand]][:topKWs]
                else:
                    cand_kw[map_rest_id2int[cand]] = result_dict[map_rest_id2int[cand]]
            else:
                cand_kw[map_rest_id2int[cand]] = []
    result_string = ', '.join(f'{key} ({", ".join(value)})' for key, value in cand_kw.items())
    return result_string


def res2kw_(res_list,new_results_res_kw, topkw = 15):
    cand_kw = {}
    if len(res_list) ==1:
        if res_list[0][0] in new_results_res_kw.keys():
            cand_kw[res_list[0][0]] = new_results_res_kw[res_list[0][0]][:topkw]
    else:
        for res in res_list:
            if res[0] in new_results_res_kw.keys():
                cand_kw[res[0]] = new_results_res_kw[res[0]][:topkw]
    if len(cand_kw) == 0:
        return 'None'
    result_string = ', '.join(f'{key} ({", ".join(value)})' for key, value in cand_kw.items())

    return result_string

def get_kw_for_rest(rest_kws, map_rest_id2int ):
    res2kw_all = dict()
    for res, kw_score in rest_kws.items():
        res2kw_all[res] = []
        for a in kw_score:
            res2kw_all[res].append(a[0])

    new_results_res_kw = {}
    for res, kws in res2kw_all.items():
        new_results_res_kw[map_rest_id2int[res]] = kws
    return new_results_res_kw

def cand_rv_fn(uid_, data, map_rest_id2int, topCandidates = 20, res_rv_ = None):
    cand_rv = {}
    for cand in data[uid_]['candidate'][: topCandidates]:
            if cand in res_rv_:
                cand_rv[map_rest_id2int[cand]] = res_rv_[cand]
    result_string = ', '.join(f'{key} ({value})' for key, value in cand_rv.items())
    return result_string

def quick_eval(preds, gt, hotel = False):
    '''
    - preds: [list of restaurants]
    - GT: [('wrdLrTcHXlL4UsiYn3cgKQ', 4.0), ('uG59lRC-9fwt64TCUHnuKA', 3.0)]
    - 
    '''
    gt_list = set([a[0] for a in gt])
    if hotel:
        gt_list = set([str(a[0]) for a in gt])

    preds_list = list(set(preds))
    ov = gt_list.intersection(preds_list)
    prec = len(ov)/len(preds_list)
    rec = len(ov)/len(gt_list)
    f1 = 0 if prec+rec == 0 else 2*prec*rec/(prec+rec)

    truth_relevant = np.asarray([[0]*len(preds)])
    if len(preds) == 1:
        ndcg = len(ov)/len(preds_list)
        return prec, rec, f1, ndcg
    for candidate in ov:
        idx = preds_list.index(candidate)
        truth_relevant[0,idx] = 1

    score = np.asarray([[x+1 for x in range(len(preds))][::-1]])
    ndcg = ndcg_score(truth_relevant, score)
    return prec, rec, f1, ndcg

def evalAll(user_rank, groundtruth, is_base = False):
    evalK = [1,3,5,10,15,20]
    for k in evalK:
        prec_final, rec_final, f1_final, ndcg_final = [],[],[], []
        for uid in user_rank.keys():
            pred =  [int(px) for px in user_rank[uid]]
            prec, rec, f1, ndcg= quick_eval(pred[:k], groundtruth[uid])
            prec_final.append(prec)
            rec_final.append(rec) 
            f1_final.append(f1) 
            ndcg_final.append(ndcg) 
        print(f'Precision@{k}: {np.mean(prec_final)}, recall@{k}: {np.mean(rec_final)}, f1@{k}: {np.mean(f1_final)} ndcg@{k}: {np.mean(ndcg_final)}')
    print(len(user_rank.keys()))