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
from data import *
import logging
import argparse
import os
from torch.utils.tensorboard import SummaryWriter 


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4'

    parser = argparse.ArgumentParser(description="training Tree-Wasserstein")
    parser.add_argument('--data_path', type=str)
    parser.add_argument("--saveloss_path", type=str)
    parser.add_argument("--savemodel_path", type=str)
    parser.add_argument("--epoch", default=3, type=int)
    parser.add_argument("--gpu", default="cuda", type=str)
    parser.add_argument("--lr", default=0.0001, type=float)    
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--n_inner", default=10, type=int)
    parser.add_argument("--n_leaf", default=4, type=int)
    parser.add_argument("--margin", default=5.0, type=float)
    
    args = parser.parse_args()

    train_dataset = MyDataset(args.data_path)
    train_loader = DataLoader(train_dataset, args.batch, shuffle=True)
    
    writer = SummaryWriter(args.savemodel_path+"/experiment")
    
    logger = get_logger(args.saveloss_path)    
    
    device = args.gpu
    
    model = modi_tree(args.n_inner, args.n_leaf, device)
        
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    logger.info('start training!')    
    for epo in range(args.epoch):

        running_loss = 0.0
        #positive_num = 0
        #negative_num = 0
        for i, sent_context_enrich in enumerate(train_loader):   #每个batch做一次参数优化（即一次正向反向）
            optimizer.zero_grad()

            loss = torch.zeros(1).to(device)
        
            
            for context_enrich in sent_context_enrich:
                

                token_emb = model(one_hot(context_enrich[0].item(),device)) 
                for positive in context_enrich[1]:
                    positive_emb = model(one_hot(positive.item(),device)) 
                    #positive_num += 1
                    #logger.info('positive number= {} '.format(positive_num))
                    pos = model.tree_loss(token_emb,positive_emb)
                    #logger.info('positive tree loss= {} '.format(pos.item()))
                    loss += pos **2
                for negative in context_enrich[2]:
                    negative_emb =  model(one_hot(negative.item(),device))
                    #negative_num += 1
                    #logger.info('negative number= {} '.format(negative_num))
                    neg = model.tree_loss(token_emb,negative_emb)
                    #logger.info('negative tree loss= {} '.format(neg.item()))
                    loss += F.relu(args.margin-neg) **2
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i%50==49:

                logger.info('Epoch:[{}]\t round:[{}]\t loss={:.5f} '.format(epo ,i , running_loss ))
                
                
                if i%500<50:
                    writer.add_scalar('loss', running_loss/50, epo * (1+int((len(train_loader)-49)/500)) + int(i/500))

                running_loss = 0.0
                
        torch.save(model.state_dict(), (args.savemodel_path + "/epoch_{}").format(epo))

    logger.info('Finished Training')
        
        
    
    
if __name__ == '__main__':
    main()

    
    

                
            
    