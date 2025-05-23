import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Patch Embedding layer for Vision Transformer.
    Splits input image into patches and projects them to embedding space.

    Args:
        img_size (int): Input image size (default: 28)
        patch_size (int): Size of each patch (default: 7)
        in_channels (int): Number of input channels (default: 1)
        embed_dim (int): Dimension of embedding (default: 256)
    """

    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Encoder block with patch embedding and positional encoding.

    Args:
        img_size (int): Input image size
        patch_size (int): Size of each patch
        in_channels (int): Number of input channels
        emb_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        dropout (float): Dropout rate
    """

    def __init__(self, img_size=28, patch_size=7, in_channels=1, emb_dim=256,
                 num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_channels, emb_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable classification token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=num_heads, dim_feedforward=emb_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # [B, N, C]
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, C]
        x = torch.cat((cls_token, x), dim=1)  # [B, N+1, C]
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.encoder(x)  # [B, N+1, C]

        return x


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


class NoisyTopkRouter(nn.Module):
    """
    Router module for Sparse Mixture of Experts.
    Routes input tokens to top-k experts.

    Args:
        input_dim (int): Input dimension
        num_experts (int): Number of experts
        k (int): Number of experts to route to
    """

    def __init__(self, input_dim, num_experts, k):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        """
        Args:
            x: [B, N, D] – input tokens
        Returns:
            gates: [B, N, E] – routing weights
            topk_indices: [B, N, k] – selected expert indices
        """
        gate_logits = self.gate(x)  # [B, N, E]
        topk_values, topk_indices = torch.topk(
            gate_logits, self.k, dim=-1)  # [B, N, k]
        topk_gates = torch.softmax(topk_values, dim=-1)  # [B, N, k]

        # Create full gate tensor with zeros, then scatter top-k values
        full_gates = torch.zeros_like(gate_logits)  # [B, N, E]
        full_gates.scatter_(-1, topk_indices, topk_gates)

        return full_gates, topk_indices


class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts module.
    Routes input tokens to multiple expert networks and combines their outputs.

    Args:
        n_embed (int): Input embedding dimension
        experts (nn.ModuleList): List of expert networks
        top_k (int): Number of experts to route to
        hidden_dim (int): Output dimension
    """

    def __init__(self, n_embed, experts, top_k, hidden_dim):
        super().__init__()
        self.router = NoisyTopkRouter(n_embed, len(experts), top_k)
        self.experts = nn.ModuleList(experts)
        self.top_k = top_k
        self.hidden_dim = hidden_dim

    def forward(self, x):
        """
        Args:
            x: [B, N, D] – input tokens
        Returns:
            [B, hidden_dim] – combined expert outputs
        """
        B, N, D = x.shape
        gating_output, indices = self.router(x)  # [B, N, E], [B, N, k]
        final_output = torch.zeros(B, self.hidden_dim).to(x.device)

        # Flatten input and gating
        flat_x = x.view(-1, D)                          # [B*N, D]
        # [B*N, E]
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):
            # Which positions (patches) go to this expert?
            expert_mask = (indices == i).any(dim=-1)  # [B, N]
            flat_mask = expert_mask.view(-1)          # [B*N]

            if flat_mask.any():
                expert_input = flat_x[flat_mask]       # [M, D]
                expert_output = expert(expert_input)   # [M, hidden_dim]
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(
                    1)  # [M, 1]

                weighted_output = expert_output * \
                    gating_scores  # [M, hidden_dim]

                # Add to final output
                batch_indices = torch.arange(B).unsqueeze(
                    1).expand(-1, N).flatten().to(x.device)
                selected_batch = batch_indices[flat_mask]  # [M]

                final_output.index_add_(0, selected_batch, weighted_output)

        return final_output


class MeOViT(nn.Module):
    """
    Vision Transformer with Mixture of Experts for classification.
    Combines ViT with patch selection and sparse MoE routing.

    Args:
        img_size (int): Input image size
        patch_size (int): Size of each patch
        in_channels (int): Number of input channels
        emb_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        dropout (float): Dropout rate
        num_classes (int): Number of output classes
        keep_ratio (float): Ratio of patches to keep
        num_experts (int): Number of experts in MoE
        top_k (int): Number of experts to route to
        expert_dim (int): Dimension of expert outputs
    """

    def __init__(self,
                 img_size=28, patch_size=7, in_channels=1,
                 emb_dim=256, num_heads=4, num_layers=3, dropout=0.1,
                 num_classes=10, keep_ratio=0.5,
                 num_experts=4, top_k=2, expert_dim=128):
        super().__init__()

        # Vision Transformer backbone
        self.vit = TransformerBlock(img_size, patch_size, in_channels, emb_dim,
                                    num_heads, num_layers, dropout)

        # Patch selection module
        self.priority = PriorityScorer(
            embed_dim=emb_dim, keep_ratio=keep_ratio)

        # Expert networks
        experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, expert_dim)
            ) for _ in range(num_experts)
        ])

        # Sparse Mixture of Experts
        self.moe = SparseMoE(n_embed=emb_dim, experts=experts,
                             top_k=top_k, hidden_dim=expert_dim)

    def forward(self, x):
        # Step 1: Process through Vision Transformer
        patch_tokens = self.vit(x)  # [B, N+1, D], N patches + CLS

        # Step 2: Remove CLS token, keep only patch tokens
        patch_tokens_only = patch_tokens[:, 1:, :]  # [B, N, D]

        # Step 3: Select important patches
        selected_x, _ = self.priority(patch_tokens_only)  # [B, K, D]

        # Step 4: Process through Sparse MoE
        moe_out = self.moe(selected_x)  # [B, expert_dim]

        return moe_out
