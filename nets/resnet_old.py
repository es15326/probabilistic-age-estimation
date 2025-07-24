
import torch
import torch.nn as nn
from loss import soft_aar_loss, weighted_mse_loss
from metrics import *

from .base import BaseNet


class ResNet(BaseNet):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.9.0", "resnet50", pretrained=pretrained
        )
        self.model.fc = nn.Linear(2048, 1)

        self.loss_func = weighted_mse_loss
        #self.loss_func = nn.MSELoss()
        # self.loss_func = soft_aar_loss
