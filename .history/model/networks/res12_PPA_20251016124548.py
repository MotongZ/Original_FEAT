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
    def __init__(self, avg_pool=True, drop_rate=0.1, dropblock_size=5):
        super().__init__()
        
        # 1. 实例化一个原始的ResNet，以便我们“借用”它的层
        # 这里传入的参数应该和你预训练时使用的参数保持一致
        original_backbone = ResNet(keep_prob=1.0, avg_pool=avg_pool, drop_rate=drop_rate, dropblock_size=dropblock_size)

        # 2. 将ResNet的四个卷积层组合成我们的编码器(encoder)
        # 这部分负责提取特征图，直到池化之前
        self.encoder = nn.Sequential(
            original_backbone.layer1,
            original_backbone.layer2,
            original_backbone.layer3,
            original_backbone.layer4,
        )

        # 3. 定义PPA模块
        # 根据你的res12.py代码，我们知道layer4的输出通道是640
        encoder_output_channels = 640
        ppa_filters = 640  # 我们让PPA的输出通道数保持一致，以便后续处理
        self.ppa_module = PPA(in_features=encoder_output_channels, filters=ppa_filters)

        # 4. 保留原始的avgpool层，保持模型行为的一致性
        self.avgpool = original_backbone.avgpool

    def forward(self, x):
        # 输入x首先通过原始ResNet的卷积层
        feature_map = self.encoder(x)
        
        # 然后，将得到的特征图送入PPA模块进行提炼
        refined_feature_map = self.ppa_module(feature_map)
        
        # 最后，进行全局平均池化和展平操作
        pooled_features = self.avgpool(refined_feature_map)
        final_features = pooled_features.view(pooled_features.size(0), -1)
        
        return final_features

def Res12_PPA(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet_PPA(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model
