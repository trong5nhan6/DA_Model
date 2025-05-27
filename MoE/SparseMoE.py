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


# ---------- Priority Scorer ----------
def compute_token_priority(gating_output, indices):
    """
    Computes priority scores for tokens based on their routing weights.

    Args:
        gating_output (torch.Tensor): Routing weights of shape [batch_size, seq_len, num_experts]
        indices (torch.Tensor): Selected expert indices of shape [batch_size, seq_len, top_k]

    Returns:
        torch.Tensor: Priority scores of shape [batch_size * seq_len]
    """
    B, N, E = gating_output.shape
    flat_gating = gating_output.view(B * N, E)
    flat_indices = indices.view(B * N, -1)
    topk_scores = torch.gather(flat_gating, 1, flat_indices)
    return topk_scores.sum(dim=1)


# ---------- Balancing Loss ----------
def balancing_loss(gating_output, alpha=0.01):
    """
    Computes the load balancing loss to ensure even distribution of tokens across experts.

    Args:
        gating_output (torch.Tensor): Routing weights of shape [batch_size, seq_len, num_experts]
        alpha (float): Scaling factor for the balancing loss

    Returns:
        torch.Tensor: Scalar balancing loss value
    """
    B, N, E = gating_output.shape
    T = B * N
    flat_gating = gating_output.view(T, E)
    p_i = flat_gating.sum(dim=0) / T
    top1 = flat_gating.argmax(dim=1)
    f_i = torch.bincount(top1, minlength=E).float() / T
    return alpha * E * (f_i * p_i).sum()


# ---------- SparseMoE ----------
class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts implementation with capacity-aware routing.

    This implementation includes:
    - Noisy top-k routing
    - Token priority-based expert assignment
    - Capacity constraints for each expert
    - Load balancing mechanism

    Args:
        n_embed (int): Input embedding dimension
        experts (List[nn.Module]): List of expert networks
        top_k (int): Number of experts to route each token to
        hidden_dim (int): Output dimension of the MoE layer
        capacity_ratio (float): Ratio of tokens each expert can process

    Input:
        x (torch.Tensor): Input tensor of shape [batch_size, seq_len, n_embed]

    Output:
        output (torch.Tensor): Output tensor of shape [batch_size, seq_len, hidden_dim]
        gating_output (torch.Tensor): Routing weights of shape [batch_size, seq_len, num_experts]
    """

    def __init__(self, n_embed, experts, top_k, hidden_dim, capacity_ratio=1.0):
        super().__init__()
        self.router = NoisyTopkRouter(n_embed, len(experts), top_k)
        self.experts = nn.ModuleList(experts)
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.capacity_ratio = capacity_ratio
        self.balance_loss = 0.0

    def forward(self, x):
        B, N, D = x.shape
        E = len(self.experts)
        device = x.device

        gating_output, indices = self.router(x)
        flat_x = x.view(-1, D)
        flat_gating_output = gating_output.view(-1, E)
        flat_indices = indices.view(-1, self.top_k)
        final_output = torch.zeros(B * N, self.hidden_dim, device=device)

        priority = compute_token_priority(gating_output, indices)
        sorted_idx = torch.argsort(priority, descending=True)

        total_tokens = B * N
        capacity_per_expert = int(
            self.capacity_ratio * self.top_k * total_tokens / E)
        token_expert_map = [[] for _ in range(E)]

        for idx in sorted_idx:
            expert_ids = flat_indices[idx]
            for eid in expert_ids:
                if len(token_expert_map[eid]) < capacity_per_expert:
                    token_expert_map[eid].append(idx.item())
                    break

        for eid, token_list in enumerate(token_expert_map):
            if not token_list:
                continue
            idx_tensor = torch.tensor(token_list, device=device)
            expert_input = flat_x[idx_tensor]
            expert_output = self.experts[eid](expert_input)
            gating_scores = flat_gating_output[idx_tensor, eid].unsqueeze(1)
            weighted_output = expert_output * gating_scores
            final_output[idx_tensor] += weighted_output

        self.balance_loss = balancing_loss(gating_output)
        return final_output.view(B, N, self.hidden_dim)
