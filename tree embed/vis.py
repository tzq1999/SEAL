import torch
import plotly.offline as plt
import plotly.graph_objs as go
from treeW import *
import numpy as np
from collections import defaultdict, Counter
import csv

class Reconstructioneval():
    def __init__(self,embedding , data,embeddim,d):
        self.embedding = embedding
        self.data = data
        self.embeddim =embeddim
        self.degree=d
                
    def embedding_distances(self,x):
        

        
        a=torch.nn.functional.softmax(self.embedding[x])
               
        for i in range(self.embeddim-1,0,-1):
            a[...,int((i-1)/self.degree)] += a[...,i]
            
            
        b=torch.nn.functional.softmax(self.embedding,dim=1)
        for i in range(self.embeddim-1,0,-1):
            b[...,int((i-1)/self.degree)] += b[...,i]
        c=torch.norm(a-b,p=1,dim=1)
        
        
        return c.numpy()



    def get_positive_relation_ranks_and_avg_prec(self,all_distances, positive_relations):

        positive_relation_distances = all_distances[positive_relations]
        negative_relation_distances = np.ma.array(all_distances, mask=False)
        negative_relation_distances.mask[positive_relations] = True
        # Compute how many negative relation distances are less than each positive relation distance, plus 1 for rank
        ranks = (negative_relation_distances < positive_relation_distances[:, np.newaxis]).sum(axis=1) + 1
        map_ranks = np.sort(ranks) + np.arange(len(ranks))
        avg_precision = ((np.arange(1, len(map_ranks) + 1) / np.sort(map_ranks)).mean())
        return list(ranks), avg_precision


    def evaluate_mean_rank_and_map(self):

        ranks = []
        avg_precision_scores = []
        for i, item in enumerate(self.data.items):
        
            if item not in self.data.positive_samples_lookup:
                continue
            item_relations = list(self.data.positive_samples_lookup[item])
        #print("item_relations",item_relations)
            for j in range(len(item_relations)):
                item_relations[j]=self.data.item2id[item_relations[j]]
       
            item_distances = self.embedding_distances(self.data.item2id[item])  # node i to all other's distance list
        #print("item_distances",item_distances)
            positive_relation_ranks, avg_precision = self.get_positive_relation_ranks_and_avg_prec(item_distances, item_relations)
            ranks += positive_relation_ranks
            avg_precision_scores.append(avg_precision)

        return np.mean(ranks), np.mean(avg_precision_scores)
    
    
class LinkPredictionEvaluation():
    """Evaluate reconstruction on given network for given embedding."""

    def __init__(self, train_path, test_path, embedding , item2id,embeddim,d):
        """Initialize evaluation instance with tsv file containing relation pairs and embedding to be evaluated.
        Parameters
        ----------
        train_path : str
            Path to tsv file containing relation pairs used for training.
        test_path : str
            Path to tsv file containing relation pairs to evaluate.
        embedding : :class:`~gensim.models.poincare.PoincareKeyedVectors`
            Embedding to be evaluated.
        """
        items = set()
        relations = {'known': defaultdict(set), 'unknown': defaultdict(set)}
        data_files = {'known': train_path, 'unknown': test_path}
        for relation_type, data_file in data_files.items():
            with open(data_file, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    assert len(row) == 2, 'Hypernym pair has more than two items'
                    item_1_index = item2id[row[0]]
                    item_2_index = item2id[row[1]]
                    relations[relation_type][item_1_index].add(item_2_index)
                    items.update([item_1_index, item_2_index])    #与上面不同，这里直接id化
                    
        self.items = items
        self.relations = relations
        self.embedding = embedding
        self.item2id = item2id

        self.embeddim =embeddim
        self.degree=d

                
    def embedding_distances(self,x):
        

        
        a=torch.nn.functional.softmax(self.embedding[x])
               
        for i in range(self.embeddim-1,0,-1):
            a[...,int((i-1)/self.degree)] += a[...,i]
            
            
        b=torch.nn.functional.softmax(self.embedding,dim=1)
        for i in range(self.embeddim-1,0,-1):
            b[...,int((i-1)/self.degree)] += b[...,i]
        c=torch.norm(a-b,p=1,dim=1)
        
        
        return c.numpy()

        
        
    @staticmethod
    def get_unknown_relation_ranks_and_avg_prec(all_distances, unknown_relations, known_relations):
        """Compute ranks and Average Precision of unknown positive relations.
        Parameters
        ----------
        all_distances : numpy.array of float
            Array of all distances for a specific item.
        unknown_relations : list of int
            List of indices of unknown positive relations.
        known_relations : list of int
            List of indices of known positive relations.
        Returns
        -------
        tuple (list of int, float)
            The list contains ranks of positive relations in the same order as `positive_relations`.
            The float is the Average Precision of the ranking, e.g. ([1, 2, 3, 20], 0.610).
        """
        unknown_relation_distances = all_distances[unknown_relations]
        negative_relation_distances = np.ma.array(all_distances, mask=False)
        negative_relation_distances.mask[unknown_relations] = True
        negative_relation_distances.mask[known_relations] = True
        # Compute how many negative relation distances are less than each unknown relation distance, plus 1 for rank
        ranks = (negative_relation_distances < unknown_relation_distances[:, np.newaxis]).sum(axis=1) + 1
        map_ranks = np.sort(ranks) + np.arange(len(ranks))
        avg_precision = ((np.arange(1, len(map_ranks) + 1) / np.sort(map_ranks)).mean())
        return list(ranks), avg_precision




     
    def evaluate_mean_rank_and_map(self, max_n=None):
        """Evaluate mean rank and MAP for link prediction.
        Parameters
        ----------
        max_n : int, optional
            Maximum number of positive relations to evaluate, all if `max_n` is None.
        Returns
        -------
        tuple (float, float)
            (mean_rank, MAP), e.g (50.3, 0.31).
        """
        ranks = []
        avg_precision_scores = []
        #print(self.items)
        #print(self.relations)
        for i, item in enumerate(self.items):
            if item not in self.relations['unknown']:  # No positive relations to predict for this node
                continue
            #print("item",item) ,item直接是index?
            unknown_relations = list(self.relations['unknown'][item])
            known_relations = list(self.relations['known'][item])
            #item_term = self.item2id[item]
            item_distances = self.embedding_distances(item)
            unknown_relation_ranks, avg_precision = \
                self.get_unknown_relation_ranks_and_avg_prec(item_distances, unknown_relations, known_relations)
            ranks += unknown_relation_ranks
            avg_precision_scores.append(avg_precision)
            if max_n is not None and i > max_n:
                break
        return np.mean(ranks), np.mean(avg_precision_scores)




if __name__ == '__main__':
    

    data = torch.load("/home/zqtan/treeW/data.pkl")
    model=PoincareEmbedding(data.n_items)
    model.load_state_dict(torch.load("/home/zqtan/treeW/epoch_200"))
    embedding=model.embedding.weight.data  #get embedding array from model 
    reconeval = Reconstructioneval(embedding,data,50,3)
    #reconeval.initialmatrix()
    #print(reconeval.matrix)
    print(reconeval.evaluate_mean_rank_and_map())
    #generate_report(model,data)
    linkpreeval=LinkPredictionEvaluation("/home/zqtan/treeW/noun_closure_part1.tsv","/home/zqtan/treeW/noun_closure_part2.tsv",embedding,data.item2id,50,3)
    #linkpreeval.initialmatrix()
    print(linkpreeval.evaluate_mean_rank_and_map())