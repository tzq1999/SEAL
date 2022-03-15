import os 
import logging
import argparse
import copy

import numpy as np
import torch

from treeW import Tree_ops, TreeWmodel, get_logger

def Linkpre_eval(model, embedding, weight, test_triple, positive_lookup, degree=2):
    
    n = len(test_triple)
    logs = []
    
    for i in range(n):
        if model.model_name == "HyperE":
            dis = HyperE(embedding, test_triple[i][0])
        elif model.model_name == "TreeWE":
            dis = TreeWE(embedding, weight, degree, test_triple[i][0])  
            
        score = np.ma.array(dis, mask=False)
        tar = copy.deepcopy(positive_lookup[test_triple[i][0]])
        tar.remove(test_triple[i][1])
        score.mask[tar] = True
        argsort = np.argsort(score)
        ranking = np.where(argsort==test_triple[i][1])[0][0]        
                    
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
        
def TreeWE(embedding, weight, degree, x):
                
    a = torch.nn.functional.softmax(embedding[x])
    embeddim = embedding.shape[1]           
    for i in range(embeddim-1,0,-1):
        a[...,int((i-1)/degree)] += a[...,i]
            
            
    b = torch.nn.functional.softmax(embedding,dim=1)
    for i in range(embeddim-1,0,-1):
        b[...,int((i-1)/degree)] += b[...,i]
        
    c = torch.norm((a-b)*weight, p=1, dim=1)
        
        
    return c.cpu().numpy()     


def HyperE(embedding, x):
    device = embedding.device
    a = [torch.norm(embedding[x])**2]*embedding.shape[0]
    a = torch.Tensor(a).to(device)
        
    b = torch.norm(embedding, dim=1)
    c = embedding -embedding[x]
    d = torch.norm(c,dim=1)

    return torch.arccosh(1+2*d**2/((1-a)*(1-b**2))).numpy()

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
    
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3, 4'

    parser = argparse.ArgumentParser(description="test model")
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--testdata_path', type=str)
    parser.add_argument('--totaldata_path', type=str)
    parser.add_argument("--save_result_path", type=str)
    parser.add_argument('--model_para_path', type=str)
    parser.add_argument("--gpu", default="cuda", type=str)  
    #parser.add_argument("--embed_dim", default=50, type=str) 
    parser.add_argument("--degree", default=2, type=str) 
        
    args = parser.parse_args()
 
    logger = get_logger(args.save_result_path) 
    logger.info(args.model_name)
    logger.info("tesing")
    
    device = args.gpu
    
    with open(args.totaldata_path,'r') as f: # every row as a pair
        raw_data = [line.strip().split('\t') for line in f.readlines()]
                
    A, B = zip(*raw_data) #get As and Bs from raw data , first and second column
        
    totalitems = tuple(set(A + B)) #individual items from dataset        
    totalitem2id = {item: i for i, item in enumerate(totalitems)} #item2id lookup table(dictionary), order is the order of csv's row

    positive_samples_lookup = {totalitem2id[a]: [] for a in A} 
    
    for a, b in raw_data:
        if totalitem2id[b] not in positive_samples_lookup[totalitem2id[a]]:    #if b not added before
            positive_samples_lookup[totalitem2id[a]].append(totalitem2id[b])
    
    test_triples = read_triple(args.testdata_path, totalitem2id)
    #all_triples = read_triple(args.totaldata_path, totalitem2id)
    
    model = TreeWmodel(args.model_name, len(totalitems), device)
    model.load_state_dict(torch.load(args.model_para_path))
    model.to(device)
    
    embedding = model.embedding.weight.data  #get embedding array from model 
    
    weight = model.weight
    
    logger.info(Linkpre_eval(model, embedding, weight, test_triples, positive_samples_lookup))
            
        
if __name__ == '__main__':
    main()
    
   
