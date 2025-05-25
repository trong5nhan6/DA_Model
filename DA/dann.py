import torch
import torch.nn as nn

# --- GRL: Gradient Reversal Layer ---


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GRL(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# --- DANN model with pluggable Feature Extractor ---
class DANN(nn.Module):
    def __init__(self, feature_extractor, feat_dim, num_classes=10, grl_lambda=1.0, label_classifier=None,
                 domain_classifier=None):
        """
        Args:
            feature_extractor (nn.Module): Any module returning [B, feat_dim]
            feat_dim (int): Dimension of features output from feature_extractor
            num_classes (int): Number of class labels
            grl_lambda (float): λ value for gradient reversal
            label_classifier (nn.Module): Optional custom label classifier
            domain_classifier (nn.Module): Optional custom domain classifier
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.label_classifier = label_classifier or nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        self.domain_classifier = domain_classifier or nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            # Binary domain prediction: source vs target
        )
        self.grl = GRL(lambda_=grl_lambda)

    def forward(self, x, alpha=1.0):
        # Expect shape [B, feat_dim]
        features = self.feature_extractor(x)
        self.grl.lambda_ = alpha
        reversed = self.grl(features)

        domain_out = self.domain_classifier(reversed)
        label_out = self.label_classifier(features)
        return label_out, domain_out
