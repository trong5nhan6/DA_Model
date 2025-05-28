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
        Forward pass for Soft-MoE.
        Args:
            x (Tensor): input tensor of shape (b, m, d)
        Returns:
            Tensor: output tensor of shape (b, d)
        """
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected x.size(-1)={x.size(-1)} to match in_features={self.in_features}, "
                f"but got {x.size(-1)}."
            )
        elif x.ndim != 3:
            raise ValueError(
                f"Expected input to have 3 dimensions, but got {x.ndim}.")

        # Compute logits and routing weights
        logits = einsum(x, self.phi, "b m d, d n p -> b m n p")
        dispatch_weights = logits.softmax(dim=1)  # D
        combine_weights = rearrange(
            logits.flatten(start_dim=2).softmax(dim=-1),
            "b m (n p) -> b m n p",
            n=self.num_experts,
        )

        # Dispatch input to experts: shape (b, n, p, d)
        x = einsum(x, dispatch_weights, "b m d, b m n p -> b n p d")

        # Apply each expert to its corresponding inputs
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            x_i = x[:, i]  # shape: (b, p, d)
            y_i = expert(x_i)  # should return (b, p, out_features)
            expert_outputs.append(y_i)

        # Stack outputs: (b, n, p, out_features)
        x = torch.stack(expert_outputs, dim=1)

        # Combine outputs
        x = einsum(x, combine_weights, "b n p d, b m n p -> b m d")  # Y
        x = x.mean(dim=1)  # hoặc torch.sum(x, dim=1)

        return x

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_experts={self.num_experts}, slots_per_expert={self.slots_per_expert}, "
            f"bias={self.bias}"
        )
