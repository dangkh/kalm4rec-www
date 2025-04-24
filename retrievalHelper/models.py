from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_
import torch.optim as optim
import torch
import json
import numpy as np
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(AutoEncoder, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    

    def get_encode(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar, notsignal)
        return z, mu, logvar


    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)
        
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

def AEloss_function(recon_x, x, mu, logvar, anneal=0.2):
    # BCE = F.binary_cross_entropy(recon_x, x)
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD

class AttentionPooling(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionPooling, self).__init__()

        # Linear layers for attention scoring
        self.V = nn.Linear(input_size, hidden_size)
        self.size = input_size
        self.w = nn.Linear(hidden_size, 1)
        self.tanh = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_features):
        # Calculate attention scores
        scores = self.tanh(self.V(input_features)) 
        scores = self.w(scores)
        
        # Apply softmax to get attention weights
        weights = self.softmax(scores)

        # Apply attention weights to input features
        pooled_features = torch.sum(weights * input_features, dim=1)

        return pooled_features, weights

class MFBPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(MFBPR, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """     
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)

        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)
        return prediction_i, prediction_j            


    def prediction(self, user, item):
        user = self.embed_user(user)
        item_i = self.embed_item(item)
        return (user * item_i).sum(dim=-1)

    def csPrediction(self, top3Users, item):
        tmpU = []
        for uid in top3Users:
            sc = self.prediction(uid, item[0])
            tmpU.append(sc.item())
        result = np.mean(np.asarray(tmpU))
        return result


def xavier_normal_initialization(module):
    r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.
    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_
    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


class JaccardSim(object):
    """docstring for JaccardSim"""
    def __init__(self, path, quantity):
        super(JaccardSim, self).__init__()
        self.path = path
        self.quantity = quantity
        f = open(path)
        keywordScore = json.load(f)
        self.rest_kw = {}
        self.l_rest = []
        for rest in keywordScore:
            self.rest_kw[rest] = []
            lw = keywordScore[rest]
            self.l_rest.append(rest)
            for kw, sc in lw:
                self.rest_kw[rest].append(kw)
        # read TFIUF contain rest: kw

        
    def pred(self, userkwList):
        sc = []
        for rest in self.rest_kw:
            sc.append(self.jscore(self.rest_kw[rest], userkwList))
        idxrest = np.argsort(sc)[::-1]
        result = [self.l_rest[x] for x in idxrest[:self.quantity]]
        return result

    def jscore(self, l1, l2):
        l1 = set(l1)
        l2 = set(l2)
        i = l1.intersection(l2)
        u = l1.union(l2)
        return len(i) / len(u)
