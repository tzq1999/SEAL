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
import logging
from torch.utils.data import Dataset,DataLoader

class MyDataset(Dataset):
    def __init__(self, path):
        

        
        corpus = Text8Corpus(path, max_sentence_length=1000)
        sentences = list(itertools.islice(Text8Corpus(path, max_sentence_length=1000), None))


        dct = Dictionary(corpus)
        dct.filter_extremes(no_below=1, no_above=0.5, keep_n=2000)  #筛选词频


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
            
            
        self.data = []
        
        
        for idx in range(len(sents2idx)):
            contexts = contexts_all[idx]
            sent = sents2idx[idx]
            contexts_enriched =[]
            for (token, context) in contexts:
                negatives = []
                for _ in range(len(context)):    #和context长度一样balanced
                    negative = random.choice(sent)  # Negative sampling happens here
                    while negative in context:
                        negative = random.choice(sent)
                    negatives.append(negative)
                contexts_enriched.append((token, context, negatives))
            self.data.append(contexts_enriched)
            
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    
    
    