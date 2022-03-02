import torch
import plotly.offline as plt
import plotly.graph_objs as go
from treeW import *
import numpy as np
from collections import defaultdict, Counter
from datasets import *
import os
from torch.utils.data import Dataset
    
    
    
def test_step(model,embedding, test_triples, all_true_triples,degree):
    
    
    device=embedding.device
    
    nentity=embedding.shape[0]
        
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
    test_dataloader_head = torch.utils.data.DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    nentity, 
                     
                    'head-batch'
                ), 
                batch_size=128,
                collate_fn=TestDataset.collate_fn
            )

    test_dataloader_tail = torch.utils.data.DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    nentity, 
                 
                    'tail-batch'
                ), 
                batch_size=128, 
                collate_fn=TestDataset.collate_fn
            )
            
    test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
    logs = []

    with torch.no_grad():
        for test_dataset in test_dataset_list:
            for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                
                positive_sample = positive_sample.to(device)
                negative_sample = negative_sample.to(device)
                filter_bias = filter_bias.to(device)

                batch_size = positive_sample.size(0)

                score = treedis((positive_sample, negative_sample), mode,embedding,degree)
                score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                argsort = torch.argsort(score, dim = 1, descending=True)

                if mode == 'head-batch':
                    positive_arg = positive_sample[:, 0]
                elif mode == 'tail-batch':
                    positive_arg = positive_sample[:, 1]
                else:
                    raise ValueError('mode %s not supported' % mode)

                for i in range(batch_size):
                            #Notice that argsort is not ranking
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking.item()
                    logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })


    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)

    return metrics


        
def treedis(sample, mode,embedding,degree):

    if mode == 'head-batch':
        tail_part, head_part = sample
        batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
        head = torch.index_select(
                embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            
        tail = torch.index_select(
                embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
        
        
    elif mode == 'tail-batch':
        head_part, tail_part = sample
        batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
        head = torch.index_select(
                embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            

            
        tail = torch.index_select(
                embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
    else:
        raise ValueError('mode %s not supported' % mode)


    #print(head.shape)
    #print(tail.shape)    
    a=torch.nn.functional.softmax(head,dim=2)
    b=torch.nn.functional.softmax(tail,dim=2)
    embeddim=embedding.shape[1]
               
    for i in range(embeddim-1,0,-1):
        a[...,int((i-1)/degree)] += a[...,i]
        b[...,int((i-1)/degree)] += b[...,i]
        
    score=-torch.norm(a-b,p=1,dim=2)
    #print(score.shape)
    
    return score
 
    
def read_triple(file_path, entity2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h,  t = line.strip().split('\t')
            triples.append((entity2id[h], entity2id[t]))
    return triples
    
    
    

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4'    
    device = "cuda"
    with open("/home/zqtan/treeW test/mammal_closure.tsv",'r') as f: # every row as a pair
        raw_data = [line.strip().split('\t') for line in f.readlines()]
        
        
    A,B = zip(*raw_data) #get As and Bs from raw data , first and second column
        
    totalitems = tuple(set(A+B)) #individual items from dataset        
    totalitem2id = {item:i for i,item in enumerate(totalitems)} #item2id lookup table(dictionary), order is the order of csv's row
    test_triples = read_triple("/home/zqtan/treeW test/mammal_closure_part2.tsv", totalitem2id)
    all_triples = read_triple("/home/zqtan/treeW test/mammal_closure.tsv", totalitem2id)
    model=PoincareEmbedding(len(totalitems))
    model.load_state_dict(torch.load("/home/zqtan/treeW test/epoch_19000"))
    model.to(device)
    embedding=model.embedding.weight.data  #get embedding array from model 
    print(test_step(model,embedding,test_triples,all_triples,3))