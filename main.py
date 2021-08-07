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
from model import *
from dataprepare import *
import logging
import argparse
import os



def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4'

    parser = argparse.ArgumentParser(description="training Tree-Wasserstein")
    parser.add_argument('--data_path', type=str)
    parser.add_argument("--saveloss_path", type=str)
    parser.add_argument("--savemodel_path", type=str)
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--gpu", default="cuda", type=str)
    parser.add_argument("--lr", default=0.0001, type=float)    
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--n_inner", default=10, type=int)
    parser.add_argument("--n_leaf", default=4, type=int)
    parser.add_argument("--margin", default=1.0, type=float)
    
    args = parser.parse_args()

    train_loader = MyDataset(args.data_path,args.batch).get()
    
    
    logger = get_logger(args.saveloss_path)    
    
    device = args.gpu
    
    model = modi_tree(args.n_inner, args.n_leaf, device)
        
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    logger.info('start training!')    
    for epo in range(args.epoch):

        running_loss = 0.0
        for i, data in enumerate(train_loader):   #每个batch做一次参数优化（即一次正向反向）
            optimizer.zero_grad()

            loss = torch.zeros(1).to(device)
        
            
            for sent in data:
                for (token, context, negatives) in sent:
	
                    token_emb = model(one_hot(token,device)) 
                    for positive in context:
                        positive_emb = model(one_hot(positive,device)) 
                    
                        loss += model.tree_loss(token_emb,positive_emb) **2
                    for negative in negatives:
                        negative_emb =  model(one_hot(negative,device)) 
                        loss += F.relu(args.margin-model.tree_loss(token_emb,negative_emb)) **2
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i%50==49:

                logger.info('Epoch:[{}]\t loss={:.5f}\t '.format(epo , running_loss ))
                running_loss = 0.0
                
        torch.save(model.state_dict(), (args.savemodel_path + "/epoch_{}").format(epo))

    logger.info('Finished Training')
        
        
    
    
if __name__ == '__main__':
    main()

    
    

                
            
    

    
    
    
    
    
