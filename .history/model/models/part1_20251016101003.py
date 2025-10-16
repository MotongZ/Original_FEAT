import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel
# Note: As in Protonet, we use Euclidean Distances here, you can change to the Cosine Similarity by replace
#       TRUE in line 30 as self.args.use_euclidean

class Part1Net(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        
    def forward(self,instance_embs, support_idx, query_idx):
        '''
        
        '''
        emb_dim = instance_embs.size(-1)
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))
        
        