
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import torch
import logging
import argparse
import os
from torch.utils.tensorboard import SummaryWriter 

from datasets import WordNetDataset
from poincare import *










def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4'

    parser = argparse.ArgumentParser(description="training hyper embed")
    parser.add_argument('--data_path', type=str)
    parser.add_argument("--saveloss_path", type=str)
    parser.add_argument("--savemodel_path", type=str)
    parser.add_argument("--epoch", default=10001, type=int)
    parser.add_argument("--gpu", default="cuda", type=str)
    parser.add_argument("--lr", default=0.03, type=float)    
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--neg_samples", default=98, type=int)
    parser.add_argument("--burn_in", default=100, type=int)
    parser.add_argument("--c", default=10, type=int)
        
    args = parser.parse_args()
 
    writer = SummaryWriter(args.savemodel_path+"/experiment")
    logger = get_logger(args.saveloss_path)    
    logger.info("Construction dataset...")
    data = WordNetDataset(filename=args.data_path,neg_samples=args.neg_samples)
    dataloader = torch.utils.data.DataLoader(data,batch_size=args.batch_size)    


    logger.info("total entries:",data.N)
    logger.info('total unique items',len(data.items))
    logger.info('negative samples per one positive',data.neg_samples,)

    torch.save(data,"/home/zqtan/hyper embed/data.pkl")



    logger.info('Training...')

    
   
    
    device = args.gpu
    
    model = PoincareEmbedding(data.n_items)
    model.to(device)
    model.initialize_embedding()       
    
    optimizer = RiemannianSGD(model.parameters())
    
    logger.info('start training!')    
    for epo in range(args.epoch):
        if epo<args.burn_in:
            lr = args.lr/args.c
        else:
            lr = args.lr

        running_loss = 0.0
        for i, batch in enumerate(dataloader):   #每个batch做一次参数优化（即一次正向反向）
            optimizer.zero_grad()

            x , y =batch
            
            preds = model(x,y)
            
            targets = torch.LongTensor([0]*preds.shape[0])
            loss= torch.nn.CrossEntropyLoss()(-preds,targets)
        
            loss.backward()
        
            optimizer.step(lr=args.lr)            
            running_loss += loss.item()
            if i%50==49:

                logger.info('Epoch:[{}]\t round:[{}]\t loss={:.5f} '.format(epo ,i , running_loss ))
                
                if i%500<50:
                    writer.add_scalar('loss', running_loss/50, epo * (1+int((len(dataloader)-49)/500)) + int(i/500))
                
                running_loss = 0.0
        
        if epo%1000==0:               
            torch.save(model.state_dict(), (args.savemodel_path + "/epoch_{}").format(epo))

    logger.info('Finished Training')
        
           
    
    
if __name__ == '__main__':
    main()

