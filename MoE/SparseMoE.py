import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- NoisyTopkRouter ----------
class NoisyTopkRouter(nn.Module):
    """
    Noisy Top-k Router for Mixture-of-Experts, adapted for input shape [B, D].

    Args:
        n_embed (int): Input embedding dimension
        num_experts (int): Total number of experts
        top_k (int): Number of experts to select per input
    """

    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.route_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [B, D]

        Returns:
            routing_weights (Tensor): [B, num_experts] (softmax over selected experts)
            indices (Tensor): [B, top_k] indices of top-k experts
        """
        logits = self.route_linear(x)  # [B, E]
        noise_std = F.softplus(self.noise_linear(x))  # [B, E]
        noise = torch.randn_like(logits) * noise_std  # [B, E]
        noisy_logits = logits + noise  # [B, E]

        # Select top-k experts for each input in the batch
        topk_logits, indices = noisy_logits.topk(
            self.top_k, dim=-1)  # [B, k], [B, k]

        # Build sparse logits with -inf for non-topk
        sparse_logits = torch.full_like(noisy_logits, float('-inf'))  # [B, E]
        # keep top-k logits only
        sparse_logits.scatter_(1, indices, topk_logits)

        # Softmax over sparse logits
        routing_weights = F.softmax(sparse_logits, dim=-1)  # [B, E]

        return routing_weights, indices  # [B, E], [B, k]


def auxiliary_loss(gating_output, indices, num_experts, beta=0.01):
    """
    Computes auxiliary loss to encourage balanced expert usage and promote diversity in routing.

    Args:
        gating_output (torch.Tensor): Routing weights of shape [batch_size, num_experts]
        indices (torch.Tensor): Selected expert indices of shape [batch_size, top_k]
        num_experts (int): Total number of experts
        beta (float): Scaling factor for the auxiliary loss

    Returns:
        torch.Tensor: Scalar auxiliary loss value
    """
    B, E = gating_output.shape
    device = gating_output.device

    # Count how many times each expert is selected (flattened over batch)
    used_experts = torch.zeros(E, device=device, dtype=torch.float32)
    used_experts.scatter_add_(
        0,
        indices.view(-1),
        torch.ones_like(indices.view(-1), dtype=torch.float32)
    )

    # Calculate expert usage ratio (how many experts were used at least once)
    expert_usage_ratio = (used_experts > 0).float().mean()

    # Compute routing entropy (encourages softmax diversity)
    routing_entropy = -torch.sum(gating_output *
                                 torch.log(gating_output + 1e-10), dim=1).mean()

    # Penalty encourages using around 50% of experts (can adjust this target if needed)
    usage_penalty = torch.abs(expert_usage_ratio - 0.5)
    # Encourage higher entropy (more uniform routing)
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
