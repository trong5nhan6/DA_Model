import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet, DenseNet


class Backbone(nn.Module):
    def __init__(self, model: nn.Module, in_channels: int = 3, img_size: tuple = (224, 224), output_dim: int = 512):
        """
        Parameters:
            model (nn.Module): Mô hình backbone (ResNet hoặc DenseNet).
            in_channels (int): Số kênh đầu vào mong muốn.
            img_size (tuple): Kích thước ảnh đầu vào (H, W).
            output_dim (int): Kích thước vector đầu ra mong muốn.
        """
        super().__init__()
        self.img_size = img_size  # (H, W)

        # Điều chỉnh lớp convolution đầu tiên nếu cần
        if isinstance(model, ResNet):
            if model.conv1.in_channels != in_channels:
                model.conv1 = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=model.conv1.out_channels,
                    kernel_size=model.conv1.kernel_size,
                    stride=model.conv1.stride,
                    padding=model.conv1.padding,
                    bias=model.conv1.bias is not None
                )
            # Loại bỏ avgpool và fc
            self.backbone = nn.Sequential(*list(model.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        elif isinstance(model, DenseNet):
            if model.features.conv0.in_channels != in_channels:
                model.features.conv0 = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=model.features.conv0.out_channels,
                    kernel_size=model.features.conv0.kernel_size,
                    stride=model.features.conv0.stride,
                    padding=model.features.conv0.padding,
                    bias=model.features.conv0.bias is not None
                )
            # Giữ lại phần feature extractor
            self.backbone = model.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        # Xác định số kênh đầu ra từ backbone
        dummy = torch.zeros(2, in_channels, *img_size)
        with torch.no_grad():
            feat = self.pool(self.backbone(dummy))
            assert feat.ndim == 4 and feat.shape[2:] == (1, 1), \
                f"Backbone must output shape [B, C, 1, 1], got {feat.shape}"
            feat_channels = feat.shape[1]

        self.proj = nn.Linear(feat_channels, output_dim)

    def forward(self, x):
        # Resize nếu cần
        if x.shape[2:] != self.img_size:
            x = F.interpolate(x, size=self.img_size,
                              mode='bilinear', align_corners=False)

        x = self.backbone(x)
        x = self.pool(x)            # [B, C, 1, 1]
        x = torch.flatten(x, 1)     # [B, C]
        x = self.proj(x)            # [B, output_dim]
        return x
