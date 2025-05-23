import torch
from Model.CNN.until import Backbone
from torchvision import models


def get_resnet18(model=models.resnet18(pretrained=True), in_channels=3, img_size=(224, 224), output_dim=512):
    resnet18_backbone = Backbone(
        model=model, in_channels=3, img_size=(224, 224), output_dim=512)
    return resnet18_backbone


def get_resnet50(model=models.resnet50(pretrained=True), in_channels=3, img_size=(224, 224), output_dim=512):
    resnet_backbone = Backbone(
        model=model, in_channels=3, img_size=(224, 224), output_dim=512)
    return resnet_backbone


def get_resnet101(model=models.resnet101(pretrained=True), in_channels=3, img_size=(224, 224), output_dim=512):
    resnet101_backbone = Backbone(
        model=model, in_channels=3, img_size=(224, 224), output_dim=512)
    return resnet101_backbone


def get_densenet121(model=models.densenet121(pretrained=True), in_channels=3, img_size=(224, 224), output_dim=512):
    densenet_backbone = Backbone(
        model=model, in_channels=3, img_size=(224, 224), output_dim=512)
    return densenet_backbone


def get_densenet161(model=models.densenet161(pretrained=True), in_channels=3, img_size=(224, 224), output_dim=512):
    densenet161_backbone = Backbone(
        model=model, in_channels=3, img_size=(224, 224), output_dim=512)
    return densenet161_backbone
