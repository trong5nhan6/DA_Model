import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Optional, Union
from einops import einsum, rearrange


class SoftMoE(nn.Module):
    """A PyTorch module for Soft-MoE, as described in:
        "From Sparse to Soft Mixtures of Experts"
        https://arxiv.org/pdf/2308.00951.pdf

    Args:
        in_features (int): input embedding dimension
        out_features (int): output embedding dimension
        experts (list[nn.Module]): list of expert modules
        slots_per_expert (int): number of slots per expert
        bias (bool): whether to include bias
        device (str or torch.device, optional)
        dtype (torch.dtype, optional)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        experts: list[nn.Module],
        slots_per_expert: int,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = len(experts)
        self.slots_per_expert = slots_per_expert
        self.bias = bias

        self.phi = nn.Parameter(
            torch.empty(
                (in_features, self.num_experts, slots_per_expert),
                device=device,
                dtype=dtype,
            )
        )
        self.experts = nn.ModuleList(experts)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward for SoftMoE with input shape [B, D]
        Returns:
            Tensor: [B, out_features]
        """
        if x.ndim != 2:
            raise ValueError(f"Expected input shape [B, D], got {x.shape}")
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected input dim {self.in_features}, got {x.size(-1)}")

        B, D = x.shape

        # Compute routing logits: [B, N, P]
        logits = einsum(x, self.phi, "b d, d n p -> b n p")

        # Compute dispatch and combine weights
        dispatch_weights = logits.softmax(dim=-1)  # softmax over slots
        combine_weights = logits.flatten(start_dim=1).softmax(
            dim=-1)  # softmax over all slots
        combine_weights = combine_weights.view(
            B, self.num_experts, self.slots_per_expert)

        # Dispatch input to experts
        # Each expert gets [B, P, D]
        expert_inputs = einsum(x, dispatch_weights, "b d, b n p -> b n p d")

        # Apply experts
        outputs = []
        for i, expert in enumerate(self.experts):
            expert_input = expert_inputs[:, i]  # [B, P, D]
            expert_output = expert(expert_input)  # [B, P, out_features]
            outputs.append(expert_output)

        # Stack all expert outputs: [B, N, P, out_features]
        outputs = torch.stack(outputs, dim=1)

        # Combine using combine weights
        y = einsum(outputs, combine_weights, "b n p d, b n p -> b d")

        return y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_experts={self.num_experts}, slots_per_expert={self.slots_per_expert}, "
            f"bias={self.bias}"
        )
