import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

TEMPLATE = np.array([[49.55128,  69.48456],[106.11472,  67.97448],[77.49944, 100.23768],[52.93336, 120.95176],[104.39072, 120.3276]])

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu(x + residual)

class Hourglass(nn.Module):
    def __init__(self, depth, f, increase=0):
        """
        depth: глубина hourglass (≥1)
        f: число каналов
        increase: увелечение числа каналов на глубине
        """
        super().__init__()
        self.depth = depth
        nf = f + increase

        self.up1 = ResidualBlock(f, f)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.low1 = ResidualBlock(f, nf)

        # Рекурсия
        if depth > 1:
            self.low2 = Hourglass(depth - 1, nf)
        else:
            self.low2 = ResidualBlock(nf, nf)

        self.low3 = ResidualBlock(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        #print(x.shape)
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        #print(up1.shape, up2.shape)
        return up1 + up2


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.conv = nn.Conv2d(x_dim, y_dim, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        #print('loss', pred.shape, gt.shape)
        #l = ((pred - gt)**2)
        #l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        #return l
        return ((pred - gt) ** 2).mean()



class LandmarksNet(nn.Module):
    def __init__(self, nstack=4, inp_dim=128, oup_dim=5, increase=0):
        super().__init__()
        """
        nstack: число HourGlass блоков
        inp_dim: число входных каналов (обычно 256)
        oup_dim: число выходных каналов (число keypoints)
        increase: увелечение числа каналов на глубине
        """

        self.nstack = nstack
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            ResidualBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, inp_dim)
        )

        self.hgs = nn.ModuleList([nn.Sequential(Hourglass(depth=3, f=inp_dim),) for i in range(nstack)])

        self.features = nn.ModuleList([nn.Sequential(
            ResidualBlock(inp_dim, inp_dim),
            nn.Conv2d(inp_dim, inp_dim, kernel_size=1)) for i in range(nstack)])

        self.outs = nn.ModuleList([nn.Conv2d(inp_dim, oup_dim, kernel_size=1) for i in range(nstack)])  # 5 heatmaps каналов
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )
        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss()

    def forward(self, x):
        #print(x.shape)
        x = self.pre(x)
        #print(x.shape)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)

    def calc_loss(self, combined_hm_preds, heatmaps):
        #combined_loss = []
        #for i in range(self.nstack):
            #combined_loss.append(self.heatmapLoss(combined_hm_preds[0][:,i], heatmaps))
            #combined_loss.append(self.heatmapLoss(combined_hm_preds[:][:, i], heatmaps))
        #combined_loss = torch.stack(combined_loss, dim=1)
        #return combined_loss
        total_loss = 0.0
        for i in range(self.nstack):
            pred = combined_hm_preds[:, i]  # [B, C, H, W]
            total_loss += self.heatmapLoss(pred, heatmaps)
        return total_loss  # уже суммировано по всем головам




# ArcFace loss module
class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.5, easy_margin=False, ls_eps=0.0):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

# Модель с ArcFace
class ArcFace_model(nn.Module):
    def __init__(self):
        super(ArcFace_model, self).__init__()
        # Используем ту же архитектуру, что и при обучении: предобученную EfficientNetB1
        self.encoding = efficientnet_b1(weights=None)  # отключаем повторное скачивание и веса
        self.bn1 = nn.BatchNorm1d(1000)
        self.arcface = ArcFace(1000, 500)

    def forward(self, x, labels=None):
        x = self.encoding(x)
        x = self.bn1(x)
        if labels is not None:
            x = self.arcface(x, labels)
        return x