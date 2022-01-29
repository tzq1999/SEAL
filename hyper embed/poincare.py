import torch
import numpy as np
import logging





class PoincareEmbedding(torch.nn.Module):
    
    def __init__(self,num_embeddings,embedding_dim=200,eps=1e-5):
        """
        Model class, which stores the embedding, passes inputs to PoincareDistance function
        
        and stores the loss fucntion.
        
        """
        super(PoincareEmbedding, self).__init__()
        
        self.eps = eps #we define the boundary to be 1-eps
        
        self.embedding = torch.nn.Embedding(num_embeddings, 
                                            embedding_dim, 
                                            padding_idx=None, 
                                            max_norm=1-self.eps, 
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
        
        self.embedding.weight.data = initial_radius*x/torch.norm(x,p=2,dim=-1).unsqueeze(-1)
        
        
    def forward(self,x,y):
        """
        Looks up the embedding given indexes and compute the distance
        """
        x_emb = self.embedding(x)
        #print("x_emb",x_emb.shape)
        y_emb = self.embedding(y)
        norm_u = torch.norm(x_emb, dim=2 )
        norm_v = torch.norm(y_emb,dim=2  )
        norm_uv = torch.norm(x_emb - y_emb ,dim=2 )
        d = 1 + 2 * norm_uv**2 / ((1 - norm_u**2) * (1 - norm_v**2) )
        dist = torch.arccosh(d)        
        
        
        
        #dist = torch.zeros((x_emb.shape[0],x_emb.shape[1]))
        #for i in range(x_emb.shape[0]):
          #for j in range(x_emb.shape[1]):
            #dist[i][j]= poincare_dist(x_emb[i][j],y_emb[i][j])
        
        
        
        return dist
    
     

class RiemannianSGD(torch.optim.Optimizer):
    """
    Mostly copied from original implementation.
    
    Riemannian stochastic gradient descent.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
    """

    def __init__(self, params):
        defaults={}
        super(RiemannianSGD, self).__init__(params,defaults)

    def poincare_grad(self,p, d_p):
        """

        Mostly copied from original implementation.

        See equation [5] in the paper.

        Function to compute Riemannian gradient from the
        Euclidean gradient in the Poincaré ball.
        Args:
            p (Tensor): Current point in the ball
            d_p (Tensor): Euclidean gradient at p
        """
    
        p_sqnorm = torch.sum(p.data ** 2, dim=-1, keepdim=True)
        d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
        d_p = torch.nan_to_num(d_p, posinf=100., neginf=-100. )
    
        return d_p

    def euclidean_retraction(self,p, d_p, lr):
        p.data.add_(d_p*(-lr))
        if torch.norm(p.data)>=1:
          p.data=(p.data/torch.norm(p.data) )-1e-5
        #p.data.add_(d_p,-lr)  #???  
        return p
        
    def step(self, lr):
        """
        Mostly copied from original implementation.
        
        Performs a single optimization step.
        Arguments:
            lr: learning rate for the current update.
        """
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                d_p = p.grad.data #gradient we computed on baclward pass 
                    
                d_p = self.poincare_grad(p, d_p)
                p = self.euclidean_retraction(p, d_p, lr)

        return loss
        
        
        
        
        
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