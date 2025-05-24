import torch
from Model.CNN.until import Backbone
from torchvision import models


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
