import torch.nn as nn
from torchvision import models

from config import EMBEDDING_SIZE, DROPOUT


class ArcFaceModel(nn.Module):
    def __init__(self, embedding_size=EMBEDDING_SIZE, pretrained=False):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.embedding = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(DROPOUT),
            nn.Linear(in_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.embedding(feat)
