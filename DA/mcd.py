import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


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
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
            # Binary domain prediction: source vs target
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
            # Binary domain prediction: source vs target
        )

        # Apply He initialization
        self._init_weights(self.classifier1)
        self._init_weights(self.classifier2)

    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
        return logits1, logits2
