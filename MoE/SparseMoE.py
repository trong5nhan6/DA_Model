import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- NoisyTopkRouter ----------
class NoisyTopkRouter(nn.Module):
    """
    A router that implements noisy top-k routing mechanism for Mixture of Experts.

    This router adds controlled noise to the routing logits to encourage exploration
    and prevent routing collapse.

    Args:
        n_embed (int): Input embedding dimension
        num_experts (int): Number of experts in the MoE layer
        top_k (int): Number of experts to route each token to

    Input:
        x (torch.Tensor): Input tensor of shape [batch_size, seq_len, n_embed]

    Output:
        routing_weights (torch.Tensor): Routing weights of shape [batch_size, seq_len, num_experts]
        indices (torch.Tensor): Selected expert indices of shape [batch_size, seq_len, top_k]
    """

    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.route_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, x):
        logits = self.route_linear(x)
        noise_std = F.softplus(self.noise_linear(x))
        noise = torch.randn_like(logits) * noise_std
        noisy_logits = logits + noise
        topk_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        mask = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = mask.scatter(-1, indices, topk_logits)
        routing_weights = F.softmax(sparse_logits, dim=-1)
        return routing_weights, indices


def auxiliary_loss(gating_output, indices, num_experts, beta=0.01):
    """
    Computes auxiliary loss to control the number of experts being used.

    Args:
        gating_output (torch.Tensor): Routing weights of shape [batch_size, seq_len, num_experts]
        indices (torch.Tensor): Selected expert indices of shape [batch_size, seq_len, top_k]
        num_experts (int): Total number of experts
        beta (float): Scaling factor for the auxiliary loss

    Returns:
        torch.Tensor: Scalar auxiliary loss value
    """
    B, N, E = gating_output.shape
    device = gating_output.device

    # Calculate number of tokens assigned to each expert
    used_experts = torch.zeros(E, device=device, dtype=torch.float32)
    used_experts.scatter_add_(
        0, indices.view(-1), torch.ones_like(indices.view(-1), dtype=torch.float32))

    # Calculate expert usage ratio
    expert_usage_ratio = (used_experts > 0).float().mean()

    # Calculate routing entropy
    flat_gating = gating_output.view(-1, E)
    routing_entropy = -torch.sum(flat_gating *
                                 torch.log(flat_gating + 1e-10), dim=1).mean()

    # Combine usage ratio and entropy penalties
    usage_penalty = torch.abs(expert_usage_ratio - 0.5)
    entropy_penalty = -routing_entropy

    return beta * (usage_penalty + entropy_penalty)


# ---------- SparseMoE ----------
class SparseMoE(nn.Module):
    """
    SparseMixture of Experts (MoE) layer.
    """

    def __init__(self, n_embed, experts, top_k, hidden_dim, beta=0.01):
        super().__init__()
        self.router = NoisyTopkRouter(n_embed, len(experts), top_k)
        self.experts = nn.ModuleList(experts)
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.auxiliary_loss = torch.tensor(0.0)

    def forward(self, x):
        """
        Input:
            x: [B, D] where D = n_embed
        Output:
            output: [B, hidden_dim]
        """
        B, D = x.shape
        gating_output, indices = self.router(x)  # [B, E], [B, k]
        final_output = torch.zeros(B, self.hidden_dim, device=x.device)

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)  # [B]
            if expert_mask.any():
                expert_input = x[expert_mask]  # [M, D]
                expert_output = expert(expert_input)  # [M, hidden_dim]
                gating_scores = gating_output[expert_mask, i].unsqueeze(
                    1)  # [M, 1]
                weighted_output = expert_output * \
                    gating_scores  # [M, hidden_dim]
                final_output[expert_mask] += weighted_output

        self.auxiliary_loss = auxiliary_loss(
            gating_output, indices, len(self.experts), self.beta)

        return final_output
