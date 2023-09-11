import math
import os
from functools import partial

import torch
import torch.nn as nn
import torchvision

from angle_prediction.loss_function import *


class AngleModel(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self):
        super(AngleModel, self).__init__()
        # self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone = torchvision.models.resnet101(pretrained=True)
        self.head = nn.Linear(1000, 3)
        self.lossf = nn.MSELoss()
        self.norm = torch.nn.BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True,
                         track_running_stats=True, device=None, dtype=None)


    def forward_encoder(self, x):
        x = self.backbone(x)
        return x

    def forward_decoder(self, x):
        x = self.norm(x)
        x = self.head(x)
        # x = nn.functional.relu(x)
        x = torch.sigmoid(x)
        # x = x*360
        return x

    def forward_loss(self, pred, label):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # print(pred.shape)
        # print(label.shape)
        pred_angle = code2angle(pred)/torch.pi*180+180
        label_code = angle2code(label)
        label_angle = label
        # print(label_code)
        # print(label)
        # print(code2angle(label_code)/torch.pi*180+180)
        # os.system("pause")
        # pred_angle = torch.zeros(pred.shape[0]).cuda()
        # pred = nn.functional.softmax(pred)
        # for i in range(pred.shape[1]):
        #     pred_angle += i*360/8*pred[:, i]
        # label_angle = torch.argmax(label, dim=1)
        pred = 2*pred - 1
        loss = self.lossf(pred, label_code)
        # print(pred_angle)
        a = torch.max(pred_angle.float(), label_angle.float())
        b = torch.min(pred_angle.float(), label_angle.float())
        acc = torch.mean(torch.min(a-b, b-a+360))
        return loss, acc

    def forward(self, imgs, labels):
        x = self.forward_encoder(imgs)
        pred = self.forward_decoder(x)
        loss, acc = self.forward_loss(pred, labels)
        pred = code2angle(pred) / torch.pi * 180 + 180
        return loss, pred, acc

