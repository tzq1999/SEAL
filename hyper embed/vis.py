import torch
import plotly.offline as plt
import plotly.graph_objs as go
from poincare import *
import numpy as np
from collections import defaultdict, Counter
import csv

class Reconstructioneval():
    def __init__(self,embedding , data):
        self.embedding = embedding
        self.data = data

    def embedding_distances(self,x):
        a=[torch.norm(self.embedding[x])**2]*self.embedding.shape[0]
        a=torch.Tensor(a)
        
        b=torch.norm(self.embedding,dim=1)
        c=self.embedding - self.embedding[x]
        d=torch.norm(c,dim=1)

        return torch.arccosh(1+2*d**2/((1-a)*(1-b**2))).numpy()



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
       
            item_distances = self.embedding_distances(i)  # node i to all other's distance list
        #print("item_distances",item_distances)
            positive_relation_ranks, avg_precision = self.get_positive_relation_ranks_and_avg_prec(item_distances, item_relations)
            ranks += positive_relation_ranks
            avg_precision_scores.append(avg_precision)

        return np.mean(ranks), np.mean(avg_precision_scores)


class LinkPredictionEvaluation():
    """Evaluate reconstruction on given network for given embedding."""

    def __init__(self, train_path, test_path, embedding , item2id):
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
                    items.update([item_1_index, item_2_index])
                    
        self.items = items
        self.relations = relations
        self.embedding = embedding
        self.item2id = item2id

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


    def embedding_distances(self,x):
        a=[torch.norm(self.embedding[x])**2]*self.embedding.shape[0]
        a=torch.Tensor(a)
        
        b=torch.norm(self.embedding,dim=1)
        c=self.embedding - self.embedding[x]
        d=torch.norm(c,dim=1)

        return torch.arccosh(1+2*d**2/((1-a)*(1-b**2))).numpy()

     
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






def generate_report(model,data): 



 
    with open("/home/zqtan/hyper embed/mammal_hierarchy.tsv",'r') as f:
        edgelist = [line.strip().split('\t') for line in f.readlines()]
        
    vis = model.embedding.weight.data.numpy()
  
  
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5,color='#888'),
        hoverinfo='none',
        mode='lines')
  
    xs = []
    ys = []
    for s0,s1 in edgelist:
        x0, y0 = vis[data.item2id[s0]]
        x1, y1 = vis[data.item2id[s1]]
  
        xs.extend(tuple([x0, x1, None]))
        ys.extend(tuple([y0, y1, None]))
  
    edge_trace['x'] = xs
    edge_trace['y'] = ys
  
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            reversescale=True,
            color='#8b9dc3',
            size=2)
        )
  
    xs = []
    ys = []
    names = []
    for name in data.items:
        x, y = vis[data.item2id[name]]
        xs.extend(tuple([x]))
        ys.extend(tuple([y]))
        names.extend(tuple([name.split('.')[0]]))
  
    node_trace['x'] = xs 
    node_trace['y'] = ys
        
    node_trace['text'] = names 
  
    display_list = ['placental.n.01',
     'primate.n.02',
     'mammal.n.01',
     'carnivore.n.01',
     'canine.n.02',
     'dog.n.01',
     'pug.n.01',
     'homo_erectus.n.01',
     'homo_sapiens.n.01',
     'terrier.n.01',
     'rodent.n.01',
     'ungulate.n.01',
     'odd-toed_ungulate.n.01',
     'even-toed_ungulate.n.01',
     'monkey.n.01',
     'cow.n.01',
     'welsh_pony.n.01',
     'feline.n.01',
     'cheetah.n.01',
     'mouse.n.01']
  
    label_trace = go.Scatter(
        x=[],
        y=[],
        mode='text',
        text=[],
        textposition='top center',
        textfont=dict(
            family='sans serif',
            size=13,
            color = "#000000"
        )
    )
  
    for name in display_list:
        x,y = vis[data.item2id[name]]
        label_trace['x'] += tuple([x])
        label_trace['y'] += tuple([y])
        label_trace['text'] += tuple([name.split('.')[0]])
  
  
  
    fig = go.Figure(data=[edge_trace, node_trace,label_trace],
                 layout=go.Layout(
                    title='Poincare Embedding of mammals subset of WordNet',
                    width=700,
                    height=700,
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
  
    plt.plot(fig, filename="/home/zqtan/hyper embed/poincare embedding.html")
  
    print('report is saves as .html files in demo folder.')

if __name__ == '__main__':
    

    data = torch.load("/home/zqtan/hyper embed/data.pkl")
    model=PoincareEmbedding(data.n_items)
    model.load_state_dict(torch.load("/home/zqtan/hyper embed/epoch_1000"))
    embedding=model.embedding.weight.data  #get embedding array from model 
    reconeval = Reconstructioneval(embedding,data)
    print(reconeval.evaluate_mean_rank_and_map())
    #generate_report(model,data)
    linkpreeval=LinkPredictionEvaluation("/home/zqtan/hyper embed/noun_closure_part1.tsv","/home/zqtan/hyper embed/noun_closure_part2.tsv",embedding,data.item2id)
    print(linkpreeval.evaluate_mean_rank_and_map())
