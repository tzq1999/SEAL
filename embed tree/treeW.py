import torch
import numpy
import logging

class my_ops(torch.autograd.Function):

    @staticmethod
    def forward(ctx, matrix, embeddim,d):
        
        matrix_=matrix.clone()
        ctx.dim = embeddim
        ctx.degree=d
        for i in range(embeddim-1,0,-1):
            matrix_[...,int((i-1)/d)] += matrix_[...,i]
        
        return matrix_

    @staticmethod
    def backward(ctx, grad_out):
        embeddim=ctx.dim
        d=ctx.degree
        grad_out_ = grad_out.clone()
        for i in range(0,int((embeddim-1)/d)):
            for j in range(1,d+1):
                grad_out_[...,d*i+j] += grad_out_[...,i]
                
        for j in range(1,embeddim-d*int((embeddim-1)/d)):
            grad_out_[...,d*int((embeddim-1)/d)+j] += grad_out_[...,int((embeddim-1)/d)]
        return grad_out_, None , None





class PoincareEmbedding(torch.nn.Module):
    
    def __init__(self,num_embeddings,embedding_dim=10,eps=1e-5,d=3):
        """
        Model class, which stores the embedding, passes inputs to PoincareDistance function
        
        and stores the loss fucntion.
        
        """
        super(PoincareEmbedding, self).__init__()
        
        self.eps = eps #we define the boundary to be 1-eps
        
        self.degree=d
        
        
        
        self.embedding_dim=embedding_dim
        
        self.embedding = torch.nn.Embedding(num_embeddings, 
                                            embedding_dim, 
                                            padding_idx=None, 
                                            max_norm=None, 
                                            norm_type=2.0, 
                                            scale_grad_by_freq=False, 
                                            sparse=False)
        

    def initialize_embedding(self,initial_radius = .001):
        """
        Initialize embedding to be a uniform disk of specified radius.
        
        The algorithm has prooven to be quite sensitive to initial state,
        
        so it would be usefull to keep it here.
        
        """
        
        distibution = torch.distributions.Uniform(-1,1)
        
        x = distibution.sample(self.embedding.weight.data.shape)
        x=x.to(self.embedding.weight.device)
        
        self.embedding.weight.data = initial_radius*x/torch.norm(x,p=2,dim=-1).unsqueeze(-1)
        
     
    
    def forward(self,x,y):
        """
        Looks up the embedding given indexes and compute the distance
        """
        x_embprob = torch.nn.functional.softmax(self.embedding(x),dim=2)
        x_embsubtreeprob=my_ops.apply(x_embprob,self.embedding_dim,self.degree)
        #print("x_emb",x_emb.shape)
        y_embprob = torch.nn.functional.softmax(self.embedding(y),dim=2)
        y_embsubtreeprob=my_ops.apply(y_embprob,self.embedding_dim,self.degree)

        tree_dis = torch.norm(x_embsubtreeprob - y_embsubtreeprob ,p=1,dim=2 )
        
        
        
        
        #dist = torch.zeros((x_emb.shape[0],x_emb.shape[1]))
        #for i in range(x_emb.shape[0]):
          #for j in range(x_emb.shape[1]):
            #dist[i][j]= poincare_dist(x_emb[i][j],y_emb[i][j])
        
        
        
        return tree_dis

    
     

        
        
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger   