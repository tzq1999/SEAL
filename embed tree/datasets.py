import numpy as np
import torch
from torch.utils.data import Dataset


class trainDataset(Dataset):
    """
    Load and parse wordnet dataset
    """

    def __init__(self,filename,neg_samples,totalitem2id):
        """
        Args:
            filename: name of .tsv file to read data from
                        Example: 'wordnet/mammal_closure.tsv'
        """
        
        #we assume data stored as A\tB in each line,
        #where A is_a B in the case of mammal closure
        #e.g. cat.n.01 is_a mammal.n.01
        
        with open(filename,'r') as f: # every row as a pair
            self.raw_data = [line.strip().split('\t') for line in f.readlines()]
        
        self.N = len(self.raw_data) #data length
        self.neg_samples = neg_samples #number of negative samples to generate 
                                       #per one positive sample
        
        A,B = zip(*self.raw_data) #get As and Bs from raw data , first and second column
        
        self.items = tuple(set(A+B)) #individual items from dataset
        
        self.n_items = len(self.items)
        
        self.item2id = totalitem2id
        
        self.positive_samples_lookup = {a:[] for a in A} #true samples lookup
                                                     #for each item, will be calculated soon
            
            # example: 'baleen_whale.n.01': ['aquatic_mammal.n.01',
            #'cetacean.n.01',
            #'placental.n.01',
            #'mammal.n.01',
            #'whale.n.02'],
    
        for a, b in self.raw_data:
            if b not in self.positive_samples_lookup[a]:    #if b not added before
                self.positive_samples_lookup[a].append(b)
        
        
        #for each positive sample we generate neg_samples negative samples
        #they are chosen randomly over all the unique items
        self.negative_samples = []
        for i in range(self.N):
            self.negative_samples.append([])
            samples_generated = 0
            while samples_generated < self.neg_samples:
                random_item = self.items[np.random.randint(self.n_items)]        
                if random_item not in self.positive_samples_lookup[self.raw_data[i][0]]:
                    self.negative_samples[-1].append(random_item)
                    samples_generated += 1
        
        
        #we vectorize data in the folloving way:
        #first element of the vector is reference item
        #second one is positive label
        #all the others are negative samples
        self.vectors = np.zeros([self.N,1+1+self.neg_samples],dtype=np.int)        
        for i in range(self.N):  # turn item to idex
            self.vectors[i][0] = self.item2id[self.raw_data[i][0]]
            self.vectors[i][1] = self.item2id[self.raw_data[i][1]]
            self.vectors[i][2:] = np.array([self.item2id[sample] for sample in self.negative_samples[i]])
    
    def __len__(self):
        return self.N

    def __getitem__(self, idx):  #input is index
        #we return list ([positive_sample,negative_sample_0,...], [reference_samples]] 
        return [self.vectors[idx][1:], self.vectors[idx][0]*np.ones(1+self.neg_samples, dtype=np.int64)]



class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, numallentity, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = numallentity
        self.mode = mode

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, tail = self.triples[idx]

        if self.mode == 'head-batch':
            
            tmp = [(0, rand_head) if (rand_head, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
