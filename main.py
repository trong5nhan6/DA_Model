import torch
from Model.CNN.until import Backbone
from Model.CNN.extract_backboone import get_resnet101
from torchvision import models
from MoE.DenseMoE import DenseMoE
from MoE.SpareMoE import MeOViT

if __name__ == "__main__":
    input_tensor = torch.randn(2, 3, 224, 224)  # Batch size 2, 3 kênh, 224x224
    input_shape = (3, 224, 224)

    # DenseMoE
    resnet18_backbone = get_resnet101()
    # Mỗi expert là một MLP nhỏ
    experts = [resnet18_backbone for _ in range(4)]

    model = DenseMoE(experts=experts, input_shape=input_shape, output_size=10)
    output = model(input_tensor)
    print(output.shape)

    # MeOViT
    model = MeOViT(img_size=224, patch_size=7, in_channels=3,
                   emb_dim=256, num_heads=4, num_layers=3, dropout=0.1,
                   num_classes=10, keep_ratio=0.5,
                   num_experts=4, top_k=2, expert_dim=128)
    output = model(input_tensor)
    print(output.shape)
