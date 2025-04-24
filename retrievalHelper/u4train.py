import random
import time
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import os
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")


def quick_eval(preds, gt, source = None, hotel = False):
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
    # if source != None :
    #     print("Precision: {}, Recall: {}, F1: {}".format(prec, rec, f1), file = source)
    return prec, rec, f1

class DataCF(Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data).type(torch.FloatTensor)
        self.label = torch.from_numpy(label).type(torch.FloatTensor)

    def __getitem__(self, index):
        '''
        return data and label
        '''
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class DataMVAE(Dataset):
    '''
    this dataset is for BPR with only embedding
    '''
    def __init__(self, label, testSet = False):
        self.label = torch.from_numpy(label)

    def __getitem__(self, index):
        '''
        return data and label
        '''
        
        return self.label[index]

    def __len__(self):
        return len(self.label)        


class DataBPR(Dataset):
    '''
    this dataset is for BPR with only embedding
    '''
    def __init__(self, label, testSet = False):
        self.label = torch.from_numpy(label)
        self.numI, self.numU = self.label.shape[1], self.label.shape[0]
        self.testSet = testSet
        # prepare for each user
        self.listU = []
        self.listR = []
        self.listNegative = []
        self.testSet = testSet
        if testSet:
            rs = [x for x in range(self.numI)]
            for userIDX in range(self.numU):
                userLB = label[userIDX]
                tmp = [userIDX] * len(userLB)
                self.listU.extend(tmp)
                self.listR.extend(rs)
        else:
            for userIDX in range(self.numU):
                userLB = label[userIDX]
                pos = np.where(userLB == 1)[0]

                tmp = [userIDX] * len(pos)
                self.listU.extend(tmp)
                self.listR.extend(pos)
                '''
                sample negative
                '''
                posN = np.where(userLB == 0)[0]
                np.random.shuffle(posN)
                numN = 0
                counter = 0
                tmp = []
                while numN < len(pos):
                    tmp.append(posN[counter])
                    numN += 1
                    counter += 1
                    counter = counter % len(posN)
                self.listNegative.extend(tmp)
            
            assert len(self.listNegative) == len(self.listU)

        assert len(self.listR) == len(self.listU)

    def __getitem__(self, index):
        '''
        return data and label
        '''
        if self.testSet:
            uidx, iidx = self.listU[index], self.listR[index]
            return   uidx,  iidx
        uidx, iidx, nidx = self.listU[index], self.listR[index], self.listNegative[index]
        # iidx = self.mapLB_Dat[iidx]
        return uidx, iidx, nidx

    def __len__(self):
        return len(self.listU)

def evaluateModel(model, data_loader, rest_train, groundtruth, users, numRetrieval, rest_Label, getPred = False):
    model.eval()
    listPred = []
    for batch_idx, (data, label) in enumerate(data_loader):
        data = data.to(device)
        label = label.to(device)
        predictions = model.prediction(data, rest_train)
        listPred.append(predictions)
    listPred = torch.vstack(listPred)
    listPred = listPred.detach().cpu().numpy()
    if getPred:
        return evaluate2pred(users, groundtruth, listPred, numRetrieval, rest_Label)
    return evaluate(users, groundtruth, listPred, numRetrieval, rest_Label)

def evaluate2pred(users, groundtruth, listPred, numRetrieval, rest_Label):
    lResults = {}
    for idx in range(len(users)):
        testUser = users[idx]
        groundtruthUser = groundtruth[testUser]
        pred = listPred[idx]
        tmp = np.argsort(pred)[::-1][:numRetrieval]
        restPred = [rest_Label[x] for x in tmp]
        # score = quick_eval(restPred, groundtruthUser)
        lResults[str(testUser)] = restPred
    return lResults

def evaluate(users, groundtruth, listPred, numRetrieval, rest_Label, city = 'singapore'):
    lResults = []
    for idx in range(len(users)):
        testUser = users[idx]
        groundtruthUser = groundtruth[testUser]
        pred = listPred[idx]
        tmp = np.argsort(pred)[::-1][:numRetrieval]
        restPred = [rest_Label[x] for x in tmp]
        score = quick_eval(restPred, groundtruthUser, None, city == 'tripAdvisor' )
        lResults.append(score)    
    return lResults    


def evaluateModel_MFBPR(model, data_loader, rest_train, groundtruth, users, numRetrieval, rest_Label):
    lResults = []
    model.eval()
    listPred = []
    for batch_idx, (userID, restID) in enumerate(data_loader):
        userID = userID.to(device)
        restID = restID.to(device)
        predictions = model.prediction(userID, restID)
        predictions = predictions.detach().cpu().numpy()
        r = [x for x in predictions]
        listPred.extend(r)
    listPred = np.asarray(listPred).reshape(len(users),-1)
    return evaluate(users, groundtruth, listPred, numRetrieval, rest_Label)


def evaluateModel_vae(model, data_loader, test_users, simU, groundtruth, users, numRetrieval, rest_Label, city = 'singapore'):
    lResults = []
    model.eval()
    trainPred = []
    listPred = []
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device).to(torch.float32)
        predictions, mu, logvar = model(data)
        predictions = predictions.detach().cpu().numpy()
        r = [x for x in predictions]
        trainPred.extend(r)
    trainPred = np.asarray(trainPred).reshape(len(users),-1)
    if len(test_users) != len(users):
        listPred = []
        for idx in range(len(test_users)):
            idu = simU[idx]
            tmp = np.sum(np.asarray([trainPred[x] for x in idu]),0)
            listPred.append(tmp)
        listPred = np.asarray(listPred)
    else:
        listPred = trainPred

    return evaluate(test_users, groundtruth, listPred, numRetrieval, rest_Label, city)
