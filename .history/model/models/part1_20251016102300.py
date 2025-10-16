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
        self.scSE = scSE(in_channel=640)
        
        
    def _forward(self,instance_embs, support_idx, query_idx):

        # organize support/query data
        
        # 应用模块整体增强 
        # enhanced_embs = self.feature_enhancer(instance_embs)
        # enhanced_embs = self.attention_module(enhanced_embs)

        emb_dim = instance_embs.size(-1)
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))
        
        proto = support.mean(dim = 1)
        
        # proto = self.prototype_refiner(proto)
        
        #复用原本逻辑
        # num_batch = proto.shape[0]
        # num_proto = proto.shape[1]
        # num_query = np.prod(query_idx.shape[-2:])



class cSE(nn.Module):

    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)  # Adding ReLU activation

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        y = self.relu(y)  # Applying ReLU activation
        y = nn.functional.interpolate(y, size=(x.size(2), x.size(3)),
                                      mode='nearest')  # Resize y to match x's spatial dimensions
        return x * y.expand_as(x)

class sSE(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, x):
        y = self.Conv1x1(x)
        y = self.norm(y)
        return x * y

class scSE(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.cSE = cSE(in_channel)
        self.sSE = sSE(in_channel)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return torch.max(U_cse, U_sse)  # Taking the element-wise maximum