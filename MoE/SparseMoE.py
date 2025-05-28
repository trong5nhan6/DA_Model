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
class PriorityScorer(nn.Module):
    """
    Module to score and select the most important patches.

    Args:
        embed_dim (int): Dimension of input embeddings
        keep_ratio (float): Ratio of patches to keep (default: 0.5)
    """

    def __init__(self, embed_dim, keep_ratio=0.5):
        super(PriorityScorer, self).__init__()
        self.keep_ratio = keep_ratio
        self.score_fn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)  # Output: scalar score per patch
        )

    def forward(self, x):
        """
        Args:
            x: [B, N, D] – patch embeddings
        Returns:
            selected_x: [B, K, D] – selected patch embeddings
            topk_indices: [B, K] – indices of selected patches
        """
        B, N, D = x.shape
        scores = self.score_fn(x).squeeze(-1)  # [B, N]
        K = int(N * self.keep_ratio)
        topk_scores, topk_indices = torch.topk(scores, k=K, dim=1)

        # Gather top-K patch embeddings
        batch_indices = torch.arange(B).unsqueeze(1).to(x.device)  # [B, 1]
        selected_x = x[batch_indices, topk_indices]  # [B, K, D]

        return selected_x, topk_indices


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
        keep_ratio (float): Ratio of patches to keep (default: 0.5)
        alpha (float): Scaling factor for the balancing loss
        beta (float): Scaling factor for the auxiliary loss

    Input:
        x (torch.Tensor): Input tensor of shape [batch_size, seq_len, n_embed]

    Output:
        output (torch.Tensor): Output tensor of shape [batch_size, seq_len, hidden_dim]
        gating_output (torch.Tensor): Routing weights of shape [batch_size, seq_len, num_experts]
    """

    def __init__(self, n_embed, experts, top_k, hidden_dim, keep_ratio=0.8, beta=0.01):
        super().__init__()
        self.router = NoisyTopkRouter(n_embed, len(experts), top_k)
        self.experts = nn.ModuleList(experts)
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.priority_scorer = PriorityScorer(n_embed, keep_ratio)
        self.beta = beta
        self.auxiliary_loss = torch.tensor(0.0)  # default value

    def forward(self, x):
        x, topk_indices = self.priority_scorer(x)
        B, N, D = x.shape
        gating_output, indices = self.router(x)  # [B, N, E], [B, N, k]
        final_output = torch.zeros(B, self.hidden_dim).to(x.device)

        flat_x = x.view(-1, D)                          # [B*N, D]
        # [B*N, E]
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)  # [B, N]
            flat_mask = expert_mask.view(-1)          # [B*N]

            if flat_mask.any():
                expert_input = flat_x[flat_mask]       # [M, D]
                expert_output = expert(expert_input)   # [M, hidden_dim]
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(
                    1)  # [M, 1]

                weighted_output = expert_output * \
                    gating_scores  # [M, hidden_dim]

                batch_indices = torch.arange(B).unsqueeze(
                    1).expand(-1, N).flatten().to(x.device)
                selected_batch = batch_indices[flat_mask]  # [M]

                final_output.index_add_(0, selected_batch, weighted_output)

        # Calculate and store auxiliary loss
        self.auxiliary_loss = auxiliary_loss(
            gating_output, indices, len(self.experts), self.beta)

        return final_output
