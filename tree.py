# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 17:40:10 2021

@author: tzq19
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
from gensim.models.word2vec import Text8Corpus
from gensim.corpora import Dictionary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4'

with open('./text8') as f:
    s = f.read()
s[:300]

corpus = Text8Corpus('./text8', max_sentence_length=1000)
sentences = list(itertools.islice(Text8Corpus('./text8', max_sentence_length=1000), None))


dct = Dictionary(corpus)
dct.filter_extremes(no_below=1, no_above=0.5, keep_n=8000)  #筛选词频
#dct.doc2idx('anarchism')

sents2idx = [dct.doc2idx(sentence) for sentence in sentences]         #编号是按照字母序对单词排的（可以通过打印看）
sents2idx = [[idx for idx in sent if idx != -1] for sent in sents2idx]
sents_8k = [[dct[idx] for idx in sent] for sent in sents2idx]
dct2 = Dictionary(sents_8k)
sents2idx = [dct.doc2idx(sentence) for sentence in sents_8k]


contexts_all = []
for sent in sents2idx:
    contexts_sent = []
    for i in range(len(sent)):
        context = []
        for j in range(max(i-2,0), min(i+2, len(sent))):
            context.append(sent[j])
        contexts_sent.append((sent[i], context))
    contexts_all.append(contexts_sent)
    
    
    
def loader(bsize):
    idx_shuffle = np.random.permutation(len(sents2idx))
    for i in range(0, len(sents2idx), bsize):
        batch = []
        for idx in idx_shuffle[i: min(i+bsize, len(sents2idx))]:
            contexts = contexts_all[idx]
            sent = sents2idx[idx]
            contexts_enriched = []
            for (token, context) in contexts:
                negatives = []
                for _ in range(len(context)):
                    negative = random.choice(sent)  # Negative sampling happens here
                    while negative in context:
                        negative = random.choice(sent)
                    negatives.append(negative)
                contexts_enriched.append((token, context, negatives))
            batch.append(contexts_enriched)
        yield batch
                
        
        
        


mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#mydevice="cpu"
        
#alpha = 2.0
               
n_inner = 10

def smoothabs(x, alpha):
    """
    differential absolute function using smooth max.

    Caution.
    if x is large (over 100), this returns nan.

    Parameters
    ----------
    x : torch.tensor
    
    Returns
    ----------
    result : torch.tensor

    """
    return  (x*torch.exp(alpha*x) - x*torch.exp(-alpha*x)) / (2.0 + torch.exp(alpha*x) + torch.exp(-alpha*x)).to(mydevice)
 


class Tree_dis(torch.nn.Module):

    def __init__(self, n_inner, n_leaf, device='cpu', max_d=None, alpha=2.0):
        super(Tree_dis, self).__init__()
        self.n_inner = n_inner # number of inner nodes which contained root.
        self.n_leaf = n_leaf # number of leaf nodes. (e.g., number of words)
        self.n_node = self.n_leaf + self.n_inner # number of all nodes.
        self.device = device
        self.alpha = alpha
               
        self.param = torch.nn.Parameter(torch.randn(self.n_inner, self.n_leaf, device=self.device)) # D2

        # initialize parameters
        nn.init.normal_(self.param, 0.0, 0.1)
        
        self.A = self.gen_A()
        self.inv_A = self.calc_inv_A() # (I - D1)^{-1}

    def gen_A(self):
        """
        Initialize D1, which is an adjacency matrix of a tree consisting of internal nodes and return I - D1.
        Assume D1 is the perfect 5-ary tree.
        """
        
        A = torch.zeros(self.n_inner, self.n_inner-1)
        for i in range(1, int(self.n_inner/5)):
            A[i-1, 5*(i-1):5*i] = 1.0
        A[int(self.n_inner/5)-1, 5*(int(self.n_inner/5)-1):] = 1.0
        A = A.to(self.device)
        return torch.eye(self.n_inner, device=self.device) - torch.cat([torch.zeros(self.n_inner, 1, device=self.device), A], dim=1).to(self.device)

    
    def calc_inv_A(self):
        """
        return (I - D1)^{-1}.
        """
        return self.A.inverse()

    
    def calc_ppar(self):
        """
        return upper two blocks of D_par.
        """
        
        exp_param = F.softmax(self.param, dim=0)
        return torch.cat([torch.eye(self.n_inner, device=self.device) - self.A, exp_param], dim=1)

    
    def calc_psub(self, block_D=None):
        """
        Retuens
        ----------
        X : torch.tensor (shape is (self.n_inner, self.n_leaf))
            X[i][j] is P_sub (v_j+self.n_inner | v_i).
        """
        
        B = F.softmax(self.param, dim=0).to(mydevice)
        X = torch.mm(self.inv_A, B).to(mydevice)
        return X.to(mydevice)

    
    def calc_distance(self, mass1, mass2, block_psub=None):
        """
        Parameters
        ----------
        mass1 : torch.tensor (shape is (self.n_leaf))
            normalized bag-of-words.
        mass2 : torch.tensor (shape is (self.n_leaf))
            normalized bag-of-words
        block_psub : torch.tensor (shape is (self.n_inner, self.n_leaf))
            retuen value of self.calc_plock_psub().
        """
        if block_psub is None:
            block_psub = self.calc_psub()
        return (smoothabs(torch.mv(block_psub, mass1 - mass2), alpha=self.alpha).sum() + smoothabs(mass1 - mass2, alpha=self.alpha).sum()).to(mydevice)

    def forward(self, x , y):
        return self.calc_distance(x,y).to(mydevice)



n_leaf = 4  #M为leaf数

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8000, 64)
        self.fc2 = nn.Linear(64, n_leaf)
        #self.param=torch.nn.Parameter(torch.randn(n_inner, n_leaf ,device=mydevice))   #D2
        #self.param=torch.randn(n_inner, n_leaf )
        self.soft= nn.Softmax()
        #self.X= calc_psub(self.param)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.soft(x)
        return x


net = Net()
net.to(mydevice)
tree_loss = Tree_dis(n_inner,n_leaf,mydevice)


optimizer = optim.Adam([obj for obj in net.parameters()]+[obj for obj in tree_loss.parameters()] ,lr=10**-4)

def one_hot(n):
    vect = torch.zeros(8000)
    vect[n] = 1.0
    return vect.to(mydevice)

m = 3
bsize = 1
epochs = 2



for epoch in range(epochs):

    running_loss = 0.0
    for i, batch in enumerate(loader(bsize), 0):   #每个batch做一次参数优化（即一次正向反向）
        optimizer.zero_grad()

        loss = torch.zeros(1).to(mydevice)
        #print("starting batch")
        for sent in batch:
            for (token, context, negatives) in sent:
                token_emb = net(one_hot(token)) 
                for positive in context:
                    positive_emb = net(one_hot(positive)) 
                    
                    loss += tree_loss(token_emb,positive_emb) **2
                for negative in negatives:
                    negative_emb =  net(one_hot(negative)) 
                    loss += net.relu(m-tree_loss(token_emb,negative_emb)) **2
            #print("finished sentence")
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i%50==49:    
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

print('Finished Training')


torch.save(net, './net.pth')

net = torch.load('./net.pth')
net.eval()

with torch.no_grad():
    test1 = net(one_hot(dct2.token2id['power']))
    test2 = net(one_hot(dct2.token2id['earth']))
    test3 = net(one_hot(dct2.token2id['mathematics']))
    test4 = net(one_hot(dct2.token2id['preservation']))
print('power',net.soft(test1))
print('earth',net.soft(test2))
print('mathematics',net.soft(test3))
print('preservation',net.soft(test4))

