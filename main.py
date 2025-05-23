import torch
from Model.CNN.until import Backbone
from Model.CNN.extract_backboone import get_resnet101
from torchvision import models
from MoE.DenseMoE import DenseMoE
from MoE.SpareMoE import MeOViT
from DA.dann import DANN_MoE
from DA.mcd import MCD

if __name__ == "__main__":
    input_tensor = torch.randn(2, 3, 224, 224)  # Batch size 2, 3 kênh, 224x224
    input_shape = (3, 224, 224)

    # DenseMoE
    resnet18_backbone = get_resnet101()
    # Mỗi expert là một MLP nhỏ
    experts = [resnet18_backbone for _ in range(4)]

    model_DenseMoE = DenseMoE(experts=experts, input_shape=input_shape)
    output_DenseMoE = model_DenseMoE(input_tensor)
    print(f"DenseMoE: {output_DenseMoE.shape}")

    # MeOViT
    model_MeOViT = MeOViT(img_size=224, patch_size=7, in_channels=3,
                   emb_dim=256, num_heads=4, num_layers=3, dropout=0.1,
                   num_classes=10, keep_ratio=0.5,
                   num_experts=4, top_k=2, expert_dim=128)
    output_MeOViT = model_MeOViT(input_tensor)
    print(f"MeOViT: {output_MeOViT.shape}")

    # DANN_MoE
    model_DANN_MoE = DANN_MoE(feature_extractor=model_MeOViT, feat_dim=128, num_classes=10, grl_lambda=1.0)
    output_DANN_MoE = model_DANN_MoE(input_tensor)
    print(f"DANN_MoE: {output_DANN_MoE}")

    # MCD
    model_MCD = MCD(feature_extractor=model_MeOViT, feat_dim=128, num_classes=10)
    output_MCD = model_MCD(input_tensor)
    print(f"MCD: {output_MCD}")