import torch
from utils import Backbone
from torchvision import models
import torch.nn as nn


def get_resnet18(in_channels=3, img_size=(224, 224), output_dim=512):
    model = models.resnet18(pretrained=True)
    resnet18_backbone = Backbone(
        model=model, in_channels=in_channels, img_size=img_size, output_dim=output_dim)
    return resnet18_backbone


def get_resnet50(in_channels=3, img_size=(224, 224), output_dim=512):
    model = models.resnet50(pretrained=True)
    resnet_backbone = Backbone(
        model=model, in_channels=in_channels, img_size=img_size, output_dim=output_dim)
    return resnet_backbone


def get_resnet101(in_channels=3, img_size=(224, 224), output_dim=512):
    model = models.resnet101(pretrained=True)
    resnet101_backbone = Backbone(
        model=model, in_channels=in_channels, img_size=img_size, output_dim=output_dim)
    return resnet101_backbone


def get_densenet121(in_channels=3, img_size=(224, 224), output_dim=512):
    model = models.densenet121(pretrained=True)
    densenet_backbone = Backbone(
        model=model, in_channels=in_channels, img_size=img_size, output_dim=output_dim)
    return densenet_backbone


def get_densenet161(in_channels=3, img_size=(224, 224), output_dim=512):
    model = models.densenet161(pretrained=True)
    densenet161_backbone = Backbone(
        model=model, in_channels=in_channels, img_size=img_size, output_dim=output_dim)
    return densenet161_backbone

# --------- Define FeatureExtractor again ----------


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),

            nn.Conv2d(50, 50, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(50 * 4 * 4, 256),
            nn.ReLU()
        )

    def forward(self, x):
        return self.backbone(x)

# --------- Load Pretrained Backbone ----------


def load_pretrained_backbone(path="Model/CNN/fashionmnist_backbone.pth"):
    model = FeatureExtractor()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model
