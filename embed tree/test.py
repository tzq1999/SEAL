   
import logging
import argparse
import os

import torch
from torch.utils.data import Dataset

from datasets import TestDataset
from treeW import Tree_ops, TreeWmodel, RiemannianSGD, get_logger   
    
    
def test_step(model, embedding, weight, test_triples, all_true_triples, degree):
        
    model.eval()
    
    device = embedding.device    
    nentity = embedding.shape[0]
        
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
                
                model_func = {
                                'TransE': TransE,
                                'HyperE': HyperE,
                                'TreeWE': TreeWE,
                                }
                
                if model.model_name =="TreeWE":
                    score = model_func[model.model_name]((positive_sample, negative_sample), mode, embedding, weight, degree)
                else:
                    score = model_func[model.model_name]((positive_sample, negative_sample), mode, embedding)               
                #score = treedis((positive_sample, negative_sample), mode,embedding,degree)
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

        
def HyperE(sample, mode, embedding):

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

    norm_u = torch.norm(head, dim=2 )
    norm_v = torch.norm(tail, dim=2  )
    norm_uv = torch.norm(head - tail ,dim=2 )
    d = 1 + 2 * norm_uv**2 / ((1 - norm_u**2) * (1 - norm_v**2) )
    score = -torch.arccosh(d)  
    
    return score
 
    
    
def TreeWE(sample, mode, embedding, weight, degree):
    #gamma = torch.tensor(2)
    #gamma = gamma.to(embedding.device)
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
    a=torch.nn.functional.softmax(head, dim=2)
    b=torch.nn.functional.softmax(tail, dim=2)
    embeddim=embedding.shape[1]
    
    for i in range(embeddim-1,0,-1):
        a[...,int((i-1)/degree)] += a[...,i]
        b[...,int((i-1)/degree)] += b[...,i]  
        
    score = -torch.norm((a-b)*weight, p=1, dim=2)
    #print(score.shape)    
    return score

def TransE(sample, mode, embedding):

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
    score=-torch.norm(head-tail,dim=2)
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
    
    device=args.gpu
    
    with open(args.totaldata_path,'r') as f: # every row as a pair
        raw_data = [line.strip().split('\t') for line in f.readlines()]
                
    A, B = zip(*raw_data) #get As and Bs from raw data , first and second column
        
    totalitems = tuple(set(A + B)) #individual items from dataset        
    totalitem2id = {item: i for i, item in enumerate(totalitems)} #item2id lookup table(dictionary), order is the order of csv's row
    
    test_triples = read_triple(args.testdata_path, totalitem2id)
    all_triples = read_triple(args.totaldata_path, totalitem2id)
    
    model = TreeWmodel(args.model_name,len(totalitems), device)
    model.load_state_dict(torch.load(args.model_para_path))
    model.to(device)
    
    embedding = model.embedding.weight.data  #get embedding array from model 
    
    weight = model.weight
    
    logger.info(test_step(model, embedding, weight, test_triples, all_triples, args.degree))
    logger.info('Finished Testing')
            
        
if __name__ == '__main__':
    main()
    