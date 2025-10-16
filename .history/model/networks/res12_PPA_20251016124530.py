import torch.nn as nn
import torch
import torch.nn.functional as F
from model.networks.dropblock import DropBlock
from res12 import ResNet

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ResNet_PPA(nn.Module):
    def __init__(self,avg_pool = True):
        super().__init__(*args, **kwargs)

def Res12_PPA(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet_PPA(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model
