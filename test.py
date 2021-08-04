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
net = torch.load('./net.pth')
net.eval()

for obj in tree_loss.parameters():
    print(obj)  