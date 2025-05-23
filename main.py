import torch
from Model.CNN.until import Backbone
from Model.CNN.extract_backboone import get_resnet101
from torchvision import models


if __name__ == "__main__":
    # resnet18 = models.resnet18(pretrained=True)

    input_tensor = torch.randn(2, 3, 224, 224)  # Batch size 2, 3 kênh, 224x224

    resnet18_backbone = get_resnet101()
    out_tensor = resnet18_backbone(input_tensor)
    # Dự kiến: [2, 512]
    print(f"ResNet18 output shape: {out_tensor.shape}")