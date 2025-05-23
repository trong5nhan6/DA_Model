import torch
import torch.nn as nn
import torch.nn.functional as F


def classifier_discrepancy(p1, p2):
    return torch.mean(torch.abs(F.softmax(p1, dim=1) - F.softmax(p2, dim=1)))


class MCD(nn.Module):
    def __init__(self, feature_extractor, feat_dim, num_classes=10):
        """
        Args:
            feature_extractor (nn.Module): Backbone model, e.g. MoE-ViT
            feat_dim (int): Output dimension of feature extractor
            num_classes (int): Number of label classes
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier1 = nn.Sequential(
            nn.Linear(feat_dim, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(feat_dim, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        """
        Returns:
            Tuple:
                - logits1: output from classifier 1
                - logits2: output from classifier 2
                - features: shared features from backbone
        """
        features = self.feature_extractor(x)
        logits1 = self.classifier1(features)
        logits2 = self.classifier2(features)
        return logits1, logits2, features
