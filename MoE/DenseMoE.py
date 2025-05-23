import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DenseMoE(nn.Module):
    def __init__(self,
                 experts,               # List[Module]
                 input_shape):          # Tuple like (C, H, W) or (features,)

        super(DenseMoE, self).__init__()

        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)

        # Flatten all dims except batch
        self.flatten = nn.Flatten(start_dim=1)

        # Compute flattened input size
        self.input_size = math.prod(input_shape)

        # Gate network to produce expert weights
        self.gate = nn.Sequential(
            nn.Linear(self.input_size, self.num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x shape: (B, C, H, W) or (B, Features)
        x_flat = self.flatten(x)  # (B, input_size)

        # (B, num_experts)
        expert_weights = self.gate(x_flat)

        # List of (B, output_size)
        # (B, num_experts, output_size)
        expert_outputs = torch.stack([expert(x)
                                     for expert in self.experts], dim=1)

        # Combine using weights: (B, output_size)
        output = torch.sum(
            expert_outputs * expert_weights.unsqueeze(-1), dim=1)

        return output
