from retrievalHelper.utils import *

class retaurantReviewG(object):
    """docstring for retaurantReviewG
    for IUF:
    keywordScore: rest: {kw: freq}
    """
    def __init__(self, args):
        super(retaurantReviewG, self).__init__()
        self.data, self.keywordScore, self.keywordFrequence, self.quantity, \
            self.edgeType, self.prohibited_groundTruth = args
        self.process()
        self.listKw_Score = None
        
    def process(self):
        self.rests = self.data['np2rests']
        # self.l_rest = []
        self.listRestCode = []
        for ii in self.rests:
            lrs = self.rests[ii]
            for r in lrs:
                if r not in self.listRestCode:
                    self.listRestCode.append(r)
        self.numRest = len(self.listRestCode)

        # self.l_user, self.user2kw= extract_users(self.data['np2users'])
        self.listUserCode, self.user2kw= extract_users(self.data['np2users'])

        self.users = self.data['np2users']
        self.kw_data = [x for x in self.data['np2count']]
        self.numUser = len(self.listUserCode)

        self.gt = {}
        for user in self.prohibited_groundTruth:
            if user in self.listUserCode:
                self.gt[user] = self.prohibited_groundTruth[user]

        self.testgt = {}
        for user in self.prohibited_groundTruth:
            if user not in self.listUserCode:
                self.testgt[user] = self.prohibited_groundTruth[user]

        self.listKW_users = {}
        for idx in self.keywordScore:
            self.listKW_users[idx] = [x[0] for x in self.keywordScore[idx]]
            
    def getInfo(self):
        print(f'number of user: {self.n_user} , number of restaurants: {self.n_rest}')

    def getKwScore(self, listkey):
        counterMissing = 0
        listScore = []
        for kw in listkey:
            value = [0]
            '''
            kwscore: user: {kw: score},..
            kiem tra tat ca cac user, neu user comment/ review = kw thi tinh trong so
            '''
            for idx in self.keywordScore:
                listKW_user = self.listKW_users[idx]
                if kw in listKW_user:
                    kwIdx = listKW_user.index(kw)
                    # cho nay them weightscore
                    value.append(self.keywordScore[idx][kwIdx][1])
            if len(value) < 2:
                counterMissing += 1
            value = np.mean(value)
            listScore.append([kw, value])
        # print(listScore)
        return listScore, counterMissing

    def getKwScore_v2(self, listkey):
        # precompute and store in self.listKw_Score
        if self.listKw_Score == None:
            # listScore = [[] for x in range(len(self.kw_data))]
            listScore = {}
            for x in self.kw_data:
                listScore[x] = []
            '''
            kwscore: rest: {kw: score},..
            kiem tra tat ca cac rest, neu rest comment/ review = kw thi tinh trong so
            '''
            for idx in self.keywordScore:
                listKW_rest = self.keywordScore[idx]
                for kw,sc in listKW_rest:
                    listScore[kw].append(sc)

            self.listKw_Score = {}
            for lsc in self.kw_data:
                self.listKw_Score[lsc] = np.mean(listScore[lsc])

        counterMissing = 0
        listScore = []
        for kw in listkey:
            value = self.listKw_Score[kw]
            listScore.append([kw, value])
        return listScore, counterMissing

    def retrievalKey(self, listkey):
        if self.keywordScore == None:
            return listkey[:self.quantity], None
        keyscore, counterMissing = self.getKwScore_v2(listkey)

        # user with keyword, score --> sort lay top score

        inds = np.argsort([x[1] for x in keyscore])
        if len(keyscore) != 1:
            inds = inds[::-1][:self.quantity]
        topK_Key = [ keyscore[x][0] for x in inds]
        keyfrequency = [float(keyscore[x][1]) for x in inds]
        return topK_Key, keyfrequency

    def retrievalKey_full(self, listkey):
        if self.keywordScore == None:
            return listkey, None
        keyscore, counterMissing = self.getKwScore_v2(listkey)
        inds = np.argsort([x[1] for x in keyscore])
        if len(keyscore) != 1:
            inds = inds[::-1]
        topK_Key = [ keyscore[x][0] for x in inds]
        keyfrequency = [float(keyscore[x][1]) for x in inds]
        return topK_Key, keyfrequency

    def retrievalKeyIdx(self, listkey):
        if self.keywordScore == None:
            return listkey[:self.quantity], None
        keyscore, counterMissing = self.getKwScore_v2(listkey)

        # user with keyword, score --> sort lay top score

        inds = np.argsort([x[1] for x in keyscore])
        inds = inds[::-1][:self.quantity]
        keyfrequency = [float(keyscore[x][1]) for x in inds]
        return inds, keyfrequency

    def checkContain(self, restid, keylist):
        tmp  = []
        for kw in keylist:
            if kw in self.rests:
                rests = self.rests[kw]
                if restid in rests:
                    return True
        return False

    def key_score2rest(self, keylist, kw_score, quantity):
        if kw_score is None:
            kw_score = np.ones(len(keylist))
        score = np.asarray(kw_score)
        fullScore = [0]*len(self.listRestCode)
        counter = 0
        for key in keylist:
            '''
            self.rest = kw: "rest: score"
            '''
            if key not in self.rests:
                continue
            listRest = self.rests[key]
            for rest in listRest:
                value = score[counter]*listRest[rest]
                restid = self.listRestCode.index(rest)
                fullScore[restid] += value
            counter += 1
        # idxrest = np.argsort(np.asarray(fullScore))
        idxrest = np.argsort(fullScore)[::-1]
        result = [self.listRestCode[x] for x in idxrest[:quantity]]
        return result

    def key_score2simUser(self, keylist, keyfrequency, numReturn = 3):
        score = np.asarray(keyfrequency)
        fullScore = [0]*len(self.listUserCode)
        restScore = [0]*len(self.listRestCode)
        counter = 0
        for key in keylist:  
            if key not in self.users:
                counter += 1
                continue
            listUser = self.users[key]
            for user in listUser:
                value = score[counter]*(listUser[user])
                userid = self.listUserCode.index(user)
                fullScore[userid] += value
            counter += 1
        '''
        --> top numReturn similar users 
        '''
        idxuser = np.argsort(fullScore)[::-1]
        result = [self.listUserCode[x] for x in idxuser[:numReturn]]
        '''
        --> top numReturn similar users 
        '''
        return result








