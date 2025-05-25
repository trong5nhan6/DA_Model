import torch
import torch.nn as nn


class DenseMoE(nn.Module):
    def __init__(self,
                 experts,      # List[nn.Module]
                 input_dim):   # int, e.g., 256 for flattened input

        super(DenseMoE, self).__init__()

        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        self.input_dim = input_dim

        # Gate to produce expert weights from flat input
        self.gate = nn.Sequential(
            nn.Linear(input_dim, self.num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x shape: (B, input_dim)
        if x.dim() > 2:
            x = x.flatten(start_dim=1)

        # Compute expert weights: (B, num_experts)
        expert_weights = self.gate(x)

        # Get expert outputs: List of (B, output_dim)
        # (B, num_experts, output_dim)
        expert_outputs = torch.stack([expert(x)
                                     for expert in self.experts], dim=1)

        # Weighted sum: (B, output_dim)
        output = torch.sum(
            expert_outputs * expert_weights.unsqueeze(-1), dim=1)

        return output
