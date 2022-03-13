import logging

import torch


class Tree_ops(torch.autograd.Function):

    @staticmethod
    def forward(ctx, matrix, embeddim, d):
        
        matrix_ = matrix.clone()
        ctx.dim = embeddim
        ctx.degree = d
        for i in range(embeddim-1, 0, -1):
            matrix_[...,int((i-1)/d)] += matrix_[...,i]
        
        return matrix_

    @staticmethod
    def backward(ctx, grad_out):
        embeddim = ctx.dim
        d = ctx.degree
        grad_out_ = grad_out.clone()
        for i in range(0, int((embeddim-1)/d)):
            for j in range(1, d+1):
                grad_out_[...,d*i+j] += grad_out_[...,i]
                
        for j in range(1, embeddim-d*int((embeddim-1)/d)):
            grad_out_[..., d*int((embeddim-1)/d)+j] += grad_out_[..., int((embeddim-1)/d)]
        return grad_out_, None , None


class TreeWmodel(torch.nn.Module):
    
    def __init__(self, model_name, num_embeddings, device, eps=1e-5, initial_radius=0.001, embedding_dim=10, d=2):
        """
        Model class, which stores the embedding, passes inputs to PoincareDistance function
        
        and stores the loss fucntion.
        
        """
        super(TreeWmodel, self).__init__()
        
        self.model_name= model_name
        self.device = device
        self.eps = eps #we define the boundary to be 1-eps
        
        self.degree = d
        
        self.num_embeddings = num_embeddings
        
        self.embedding_dim = embedding_dim
        
        #self.gamma = gamma
        
        self.weight = torch.ones(embedding_dim).to(device)
        
        if self.model_name== "HyperE":
            self.embedding = torch.nn.Embedding(self.num_embeddings, 
                                            self.embedding_dim, 
                                            padding_idx=None, 
                                            max_norm=1-self.eps, 
                                            norm_type=2.0, 
                                            scale_grad_by_freq=False, 
                                            sparse=False)            
            
            distibution = torch.distributions.Uniform(-1,1)
        
            x = distibution.sample(self.embedding.weight.data.shape)
            x = x.to(device)
        
            self.embedding.weight.data = initial_radius*x/torch.norm(x,p=2,dim=-1).unsqueeze(-1)            
            
        else:
            self.embedding = torch.nn.Embedding(self.num_embeddings, 
                                            self.embedding_dim, 
                                            padding_idx=None,  
                                            norm_type=2.0, 
                                            scale_grad_by_freq=False, 
                                            sparse=False)
            distibution = torch.distributions.Uniform(-1,1)
        
            x = distibution.sample(self.embedding.weight.data.shape)
            x = x.to(device)
        
            self.embedding.weight.data = initial_radius*x/torch.norm(x,p=2,dim=-1).unsqueeze(-1)
             
    
    def forward(self, x, y):
        """
        Looks up the embedding given indexes and compute the distance
        """
            
        model_func = {
            'TransE': self.TransE,
            'HyperE': self.HyperE,
            'TreeWE': self.TreeWE,
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](x,y)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score    
     
    def TransE(self, x, y):

        score = -torch.norm(self.embedding[x]-self.embedding[y],dim=2)
        return score

    def HyperE(self, x, y):
        
        x_emb = self.embedding(x)        
        y_emb = self.embedding(y)
        norm_u = torch.norm(x_emb, dim=2 )
        norm_v = torch.norm(y_emb,dim=2  )
        norm_uv = torch.norm(x_emb - y_emb ,dim=2 )
        d = 1 + 2 * norm_uv**2 / ((1 - norm_u**2) * (1 - norm_v**2) )
        score = -torch.arccosh(d)        
        return score

    def TreeWE(self, x, y):
        
        x_embprob = torch.nn.functional.softmax(self.embedding(x),dim=2)
        x_embsubtreeprob = Tree_ops.apply(x_embprob,self.embedding_dim,self.degree)
        #print("x_emb",x_emb.shape)
        y_embprob = torch.nn.functional.softmax(self.embedding(y),dim=2)
        y_embsubtreeprob = Tree_ops.apply(y_embprob,self.embedding_dim,self.degree)
        
        score = -torch.norm((x_embsubtreeprob - y_embsubtreeprob)*self.weight ,p=1,dim=2 )
        return score
         
class RiemannianSGD(torch.optim.Optimizer):
    """
    Mostly copied from original implementation.
    
    Riemannian stochastic gradient descent.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
    """

    def __init__(self, params, lr):
        defaults = {}
        super(RiemannianSGD, self).__init__(params,defaults)
        self.lr = lr

    def poincare_grad(self, p, d_p):
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

    def euclidean_retraction(self, p, d_p, lr):
        p.data.add_(d_p*(-lr))
        if torch.norm(p.data)>=1:
          p.data=(p.data/torch.norm(p.data) )-1e-5
        #p.data.add_(d_p,-lr)  #???  
        return p
        
    def step(self):
        """
        Mostly copied from original implementation.
        
        Performs a single optimization step.
        Arguments:
            lr: learning rate for the current update.
        """
        loss = None
        lr=self.lr

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