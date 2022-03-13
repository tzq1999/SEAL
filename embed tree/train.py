import logging
import argparse
import os

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter 

from datasets import trainDataset
from treeW import Tree_ops, TreeWmodel, RiemannianSGD, get_logger


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4'
    parser = argparse.ArgumentParser(description="training model")
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--testdata_path', type=str)
    parser.add_argument('--traindata_path', type=str)
    parser.add_argument("--saveloss_path", type=str)
    parser.add_argument("--savemodel_path", type=str)
    parser.add_argument("--epoch", default=10001, type=int)
    parser.add_argument("--gpu", default="cuda", type=str)
    parser.add_argument("--lr", default=0.001, type=float)    
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--neg_samples", default=10, type=int)
        
    args = parser.parse_args()
 
    writer = SummaryWriter(args.savemodel_path+"/experiment")
    logger = get_logger(args.saveloss_path)    
    logger.info(args.model_name)
    logger.info("Construction dataset...")
        
    with open(args.data_path,'r') as f: # every row as a pair
        raw_data = [line.strip().split('\t') for line in f.readlines()]
               
    A, B = zip(*raw_data) #get As and Bs from raw data , first and second column
        
    totalitems = tuple(set(A + B)) #individual items from dataset    
    totalnum=len(totalitems)        
    totalitem2id = {item:i for i,item in enumerate(totalitems)} #item2id lookup table(dictionary), order is the order of csv's row
        
    #test = WordNetDataset(filename=args.testdata_path,neg_samples=args.neg_samples)
    train = trainDataset(args.traindata_path,args.neg_samples,totalitem2id)
    dataloader = torch.utils.data.DataLoader(train,batch_size=args.batch_size,shuffle=True)    
    
    #logger.info("total train entries:",train.N)
    #logger.info('total unique items in train',len(train.items))
    #logger.info('negative samples per one positive',train.neg_samples,)

    #torch.save(data,args.savemodel_path+"/data.pkl")
    logger.info('Training...')
    
    device = args.gpu
    
    model = TreeWmodel(args.model_name,totalnum,device)
    #model.to(device)
    #model.initialize_embedding()
    model.train()
    
    if args.model_name=="HyperE":
        optimizer=RiemannianSGD(model.parameters(), lr=args.lr)        
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    logger.info('start training!')
      
    for epo in range(args.epoch):

        running_loss = 0.0
        
        for i, (x, y) in enumerate(dataloader):   #每个batch做一次参数优化（即一次正向反向）
            optimizer.zero_grad()

            x , y = x.to(device) , y.to(device)           
                        #print(x.shape)
            #print(x,y)
            preds = model(x,y)
            #print(preds.shape)
            targets = torch.LongTensor([0]*preds.shape[0]).to(device)           
            loss= torch.nn.CrossEntropyLoss(reduction="sum")(preds,targets)        
            loss.backward()
            #torch.nn.utils.clip_grad_value_(model.parameters(),clip_value=1.2)        
            optimizer.step()            
            running_loss += loss.item()
            if i%30==29 and epo%100==0:

                logger.info('Epoch:[{}]\t round:[{}]\t eachroundloss={:.5f} '.format(epo ,i , running_loss/30 ))
                
                if i%500<50:
                    writer.add_scalar('loss', running_loss/30, epo * (1+int((len(dataloader)-49)/500)) + int(i/500))
                
                running_loss = 0.0
        
        if epo%100==0:               
            torch.save(model.state_dict(), (args.savemodel_path + "/epoch_{}").format(epo))

    logger.info('Finished Training')
            
    
if __name__ == '__main__':
    main()

