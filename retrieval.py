import argparse
from ast import parse
import os
import time
from retrievalHelper.utils import *
from reRanker.utils import *
from kwExtractorHelper.utils import mkdir
from retrievalHelper.u4Res import *
from retrievalHelper.u4KNN import *
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")

random_seed = 1001
setSeed(random_seed)

listcity = ['charlotte', 'edinburgh', 'lasvegas', 'london', 'phoenix', 'pittsburgh', 'singapore']

parser = argparse.ArgumentParser()
parser.add_argument('--city', type=str, default='edinburgh', help=f'choose city{listcity}')
parser.add_argument('--quantity', type=int, default=20, help='number of keyword retrieval')
parser.add_argument('--seed', type=int, default=1001, help='number of keyword retrieval')
parser.add_argument('--edgeType', type=str, default='IUF', help='weight score for keyword')

'''
Export args
'''
parser.add_argument('--logResult', type=str, default='./log', help='write log result detail')
parser.add_argument('--export2LLMs', action='store_true', help='whether export list of data for LLMs or not. \
                                                                default = False')

'''
Model args
'''
parser.add_argument('--RetModel', type=str, default='MPG', help='Jaccard, MF, MVAE, CLCp, MPG_old, MPG')
parser.add_argument('--numKW4FT', type=int, default=20, help='number of keyword for feature')
parser.add_argument('--validTopK', type=int, default=20, help='number of keyword for feature')


args = parser.parse_args()

print("args:", args)
print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print("Loading training keyword")
trainDat, testDat = data_reviewLoader(args.city)
train_users, train_users2kw = extract_users(trainDat['np2users'])
test_users, test_users2kw = extract_users(testDat['np2users'])

# extract user2rest for label
gt = load_groundTruth(f'./data/reviews/{args.city}.csv')
# load edgeType


keywordScore, keywordFrequence = load_kwScore(args.city, args.edgeType)
restGraph = retaurantReviewG([trainDat, keywordScore, keywordFrequence, \
                                args.quantity,  args.edgeType, gt])

KNN = neighbor4kw(f'{args.city}_kwSenEB_pad', testDat,  restGraph)
rest_Label = getRestLB(trainDat['np2rests'])
sourceFile = open(args.logResult, 'a')


print('*'*10, 'Result' ,'*'*10, file = sourceFile)
prediction = []
numUser = len(train_users)
numItem = len(rest_Label)

print("*"*50)
print("using MPG")
print("*"*50)
l_rest = restGraph.listRestCode
kw_data = KNN.kw_data
lu, li, lh = list([]), list([]), list([])
for rest in tqdm(keywordScore):
    kw_scs = keywordScore[rest]
    u = l_rest.index(rest)
    tmpI, tmpH = [], []
    for kw, sc in kw_scs:
        tmpI.append(kw_data.index(kw))
        tmpH.append(sc)
    tmpU = [u] * len(tmpI)
    lu.extend(tmpU)
    li.extend(tmpI)
    lh.extend(tmpH)

adj = np.zeros([len(l_rest), len(kw_data)])
for ii in range(len(lu)):
    u, v, w = lu[ii], li[ii], lh[ii]
    if args.edgeType == "IUF":
        adj[u, v] = w
    else:
        adj[u, v] = 1

lResults = []
lidx = [x for x in range(len(test_users))]
np.random.shuffle(lidx)
dictionary = {}
listsimU = []
if args.city=='tripAdvisor':
    tmpHelper = regionHelper(l_rest)
for ite in tqdm(range(len(test_users))):
    idx = lidx[ite]
    testUser, topK_Key, keyfrequency, topUser = procesTest(test_users, test_users2kw, idx, KNN, restGraph, args.export2LLMs)
    testkey = [kw_data.index(x) for x in topK_Key]
    ft = np.zeros(len(kw_data))
    for x in testkey: ft[x] = 1.0
    ft = ft.reshape(-1, 1)
    tmp = np.matmul(adj, ft).reshape(-1)
    if args.city=='tripAdvisor':
        sc = tmpHelper.query(testUser)
        tmp = tmp*sc
    idxrest = np.argsort(tmp)[::-1]
    result = [l_rest[x] for x in idxrest[:args.quantity]]
    prediction.append(result)
    groundtruth = gt[testUser]
    score = quick_eval(result[:args.validTopK], groundtruth, args.city=='tripAdvisor')
    lResults.append(score)
    if args.export2LLMs:
        simU = topUser
        userInteract = {'kw':topK_Key, 'score_kw': keyfrequency, 'candidate':result, 'simUser': simU}
        listsimU.extend(simU)
        dictionary[testUser] = userInteract
if args.export2LLMs:
    trainUwithCandi = {}
    listsimU = list(set(train_users))
    # listsimU = list(set(listsimU))
    for idx in tqdm(range(len(listsimU))):
        userIdx = train_users.index(listsimU[idx])
        user_key = train_users2kw[userIdx]
        topK_Key, keyfrequency = restGraph.retrievalKey(user_key)
        tmp = {}
        for key, sc in zip(topK_Key, keyfrequency):
            tmp[key] = sc
        trainUwithCandi[listsimU[idx]] = tmp

    json_object = json.dumps(trainUwithCandi, indent=4)
    mkdir("./data/out2LLMs/")
    with open(f"./data/out2LLMs/{args.city}_user2candidate.json", "w") as outfile:
        outfile.write(json_object)
    json_object = json.dumps(dictionary, indent=4)
    with open(f"./data/out2LLMs/{args.city}_knn2rest.json", "w") as outfile:
        outfile.write(json_object) 
p, r, f, n = extractResult(lResults)

print("args:", args, file = sourceFile)
k = args.validTopK
print(f'Pre@{k}: {mean(p)}, Recall@{k}:{mean(r)}, F1@{k}:{mean(f)}, NDCG@{k}:{mean(n)}', file = sourceFile)
print('*'*10, 'End' ,'*'*10, file = sourceFile)
sourceFile.close()
print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))


















