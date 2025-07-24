
import efficientnet_pytorch as enp
import torch
import torch.nn as nn
from loss import soft_aar_loss, weighted_mse_loss
from metrics import *

from .base import BaseNet


class EfficientNet(BaseNet):
    def __init__(self, pretrained='efficientnet-b7'):
        super().__init__()
        self.model = enp.EfficientNet.from_pretrained(pretrained, num_classes=1)
        # self.loss_func = weighted_mse_loss
        #self.loss_func = nn.MSELoss()
        self.loss_func = soft_aar_loss
