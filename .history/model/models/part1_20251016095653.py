import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel
from model.models.protonet import ProtoNet
# Note: As in Protonet, we use Euclidean Distances here, you can change to the Cosine Similarity by replace
#       TRUE in line 30 as self.args.use_euclidean

class Part1Net(ProtoNet):
    def __init__(self, args):
        super().__init__(args)
        
    def forward(self,data_shot,data_query):
        