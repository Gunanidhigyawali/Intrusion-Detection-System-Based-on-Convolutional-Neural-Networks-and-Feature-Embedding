import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ARCFACE_S, ARCFACE_M


class ArcMarginProduct(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss  (https://arxiv.org/abs/1801.07698)
    Adds a fixed angular margin m to the target class angle before softmax.
    """

    def __init__(self, in_features, out_features, s=ARCFACE_S, m=ARCFACE_M):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)   # cos(π - m)
        self.mm = math.sin(math.pi - m) * m       # fallback shift

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine   = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=1e-6))

        phi = cosine * self.cos_m - sine * self.sin_m          # cos(θ + m)
        # Stable fallback when θ + m > π
        phi = torch.where(cosine > self.threshold, phi,
                          cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s
