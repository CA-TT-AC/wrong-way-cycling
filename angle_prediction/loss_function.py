import math

import torch
import torch.nn as nn


def angle2code(phi):
    phi = phi / 360 * 2 * torch.pi - torch.pi
    ans = torch.zeros(phi.shape[0], 3).to('cuda:0')
    for i in range(3):
        ans[:, i] = torch.cos(phi + 2 * (i+1) * torch.pi / 3)
    return ans


def code2angle(x):
    sum_up = torch.zeros(x.shape[0]).to('cuda:0')
    sum_down = torch.zeros(x.shape[0]).to('cuda:0')
    for i in range(3):
        sum_up = sum_up + x[:, i] * math.sin(2 * (i+1) * torch.pi / 3)
        sum_down = sum_down + x[:, i] * math.cos(2 * (i+1) * torch.pi / 3)
    ans = -torch.atan2(sum_up, sum_down)
    return ans


class MultiBinLoss(nn.Module):
    def __init__(self, num_bins=8):
        super().__init__()
        self.num_bins = num_bins
        self.bin_width = 360 / num_bins
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, y_pred, y_true):
        # 把角度映射到 [0, 2π] 区间内
        bin_idxs = (y_true / self.bin_width).long()

        # 计算每个样本的 softmax 损失
        bin_probs = self.softmax(y_pred)
        bin_losses = torch.zeros_like(bin_probs)
        for i in range(self.num_bins):
            mask = (bin_idxs == i)
            if mask.sum() > 0:
                bin_losses[mask] = -torch.log(bin_probs[mask])

        # 计算最终的损失值
        loss = bin_losses.sum(dim=-1).mean()
        return loss
