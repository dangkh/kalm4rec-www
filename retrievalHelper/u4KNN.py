import json
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from retrievalHelper.utils import *

class neighbor4kw(object):
    """docstring for neighbor4kw"""
    def __init__(self, path, testDat, restGraph):
        super(neighbor4kw, self).__init__()
        self.path = path
        self.kwEB_pad = np.load(f"./data/embedding/{path}_train.npy")

        self.neigh = NearestNeighbors(n_neighbors=1)
        self.neigh.fit(self.kwEB_pad)
        self.kwEB_pad_test = np.load(f"./data/embedding/{path}_test.npy")
        self.testDat = testDat
        self.test_users, self.test_users2kw = extract_users(testDat['np2users'])
        self.kw_testData = [x for x in testDat['np2count']]
        self.restGraph = restGraph
        self.trainDat, self.keywordScore = restGraph.data, restGraph.keywordScore
        self.kw_data = restGraph.kw_data
        self.rest_train_data =  restGraph.rests

    def get_topK_Key(self, testkey):
        lEkey = []
        for tk in testkey:
            lEkey.append(self.get_testEmb(tk))
        knn = self.getKneighbors(lEkey)
        topK_Key = []
        keyfrequency = []
        for x, y in zip(knn[1], knn[0]):
            topK_Key.append(self.kw_data[x[0]])
            # keyfrequency.append(max(1 - max(y[0],1000)/1000, 0))
        return topK_Key


    def getKneighbors(self, lEkey):
        return self.neigh.kneighbors(lEkey)


    def get_testEmb(self, tk):
        pos = self.kw_testData.index(tk)
        return self.kwEB_pad_test[pos]

    def loadFT(self, numKW4FT, rest_Label, city):
        
        assert self.kwEB_pad.shape[0] == len(self.kw_data)
        # extract user feature

        userKw = []
        for uid, user in enumerate(tqdm(self.restGraph.listUserCode)):
            kws = self.restGraph.user2kw[uid]
            topK_Key, _ = self.restGraph.retrievalKey(kws)
            counter = 0
            while len(topK_Key) < numKW4FT:
                topK_Key.append(topK_Key[counter])
                counter += 1
            userKw.append(topK_Key)

        userFT = userKW2FT(userKw, self.kwEB_pad, self.kw_data)
        userFT = np.asarray(userFT)


        restFT = []
        rest_kw = []
        rest_ALL = []
        rest_train = []
        rest_kw_final = []
        
        
        f = open(f'./data/score/{city}-keywords-TFIUF.json')
        keywordScoreRest = json.load(f)
        for restIdx, rest in enumerate(keywordScoreRest):
            listrest_kw = keywordScoreRest[rest]
            rest_ALL.append(rest)
            restFT.append([])
            rest_kw.append([])
            for kw, score in listrest_kw:
                emID = self.kw_data.index(kw)
                emKW = self.kwEB_pad[emID]
                restFT[-1].append(emKW)
                rest_kw[-1].append(kw)
        for idx in range(len(restFT)):
            counter = len(restFT[idx])
            while counter < numKW4FT * 3:
              counter += 1
              restFT[idx].append(np.zeros(384))
              rest_kw[idx].append(rest_kw[idx][0])
            rest_train.append(restFT[idx][: numKW4FT * 3])
            rest_kw_final.append(rest_kw[idx][:numKW4FT * 3])
        rest_train = np.asarray(rest_train)


        '''
        compute rest map
        given i pos in restLabel, which corresponding index in restFeature
        '''
        rest_map = [rest_ALL.index(x) for x in rest_Label]
        tmp = np.zeros_like(rest_train)

        for ii in range(len(rest_train)):
            tmp[rest_map[ii]] = rest_train[ii]

        rest_train = tmp

        userFT_test = []
        for idx, user in enumerate(self.test_users):
            testkey = self.test_users2kw[idx]
            testkey = self.get_topK_Key(testkey)

            topK_Key, keyfrequency = self.restGraph.retrievalKey(testkey)
            ft = []
            for ii in range(min(len(topK_Key), numKW4FT)):
                emID = self.kw_data.index(topK_Key[ii])
                ft.append(self.kwEB_pad[emID])
            while (len(ft) < numKW4FT):
                ft.append(np.zeros(384))
            userFT_test.append(ft)

        userFT_test = np.asarray(userFT_test)

        return userFT, userFT_test, rest_train

    def loadrawKWs(self, numKW4FT, rest_Label, city):
        
        userkw_test = []
        for idx, user in enumerate(tqdm(self.test_users)):
            testkey = self.test_users2kw[idx]
            testkey = self.get_topK_Key(testkey)

            topK_Key, keyfrequency = self.restGraph.retrievalKey(testkey)
            counter = 0
            while (len(topK_Key) < numKW4FT):

                topK_Key.append(topK_Key[counter])
                counter += 1
            userkw_test.append(topK_Key)

        return userkw_test

   