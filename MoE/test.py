import torch
import torch.nn as nn
from torch import einsum
from SoftMoE import SoftMoE
import math

# === SoftMoE class (bạn đã định nghĩa sẵn) ===
# Dán lớp SoftMoE ở đây nếu chưa import

# === Test Script ===

# Config
in_features = 4
out_features = 2
num_experts = 3
slots_per_expert = 2
batch_size = 2

# Tạo các expert đơn giản
experts = [nn.Sequential(
    nn.Linear(in_features, out_features),
    nn.ReLU()
) for _ in range(num_experts)]

# Khởi tạo mô hình
model = SoftMoE(
    in_features=in_features,
    out_features=out_features,
    experts=experts,
    slots_per_expert=slots_per_expert
)

# Đầu vào ngẫu nhiên
x = torch.randn(batch_size, in_features)

# === Hook logic để theo dõi biến trung gian ===


def forward_intercept(model, x):
    B, D = x.shape

    phi = model.phi  # [D, N, P]
    print("phi:", phi.shape)
    print(phi)

    # Tính logits
    logits = torch.einsum("b d, d n p -> b n p", x, phi)
    print("logits:", logits.shape)
    print(logits)

    # Dispatch weights
    dispatch_weights = logits.softmax(dim=-1)  # [B, N, P]
    print("dispatch_weights:", dispatch_weights.shape)
    print(dispatch_weights)

    # Combine weights
    combine_weights = logits.flatten(start_dim=1).softmax(dim=-1)
    combine_weights = combine_weights.view(
        B, model.num_experts, model.slots_per_expert)
    print("combine_weights:", combine_weights.shape)
    print(combine_weights)

    # Expert inputs: [B, N, P, D]
    expert_inputs = torch.einsum("b d, b n p -> b n p d", x, dispatch_weights)
    print("expert_inputs:", expert_inputs.shape)
    print(expert_inputs)

    # Expert outputs
    outputs = []
    for i, expert in enumerate(model.experts):
        out = expert(expert_inputs[:, i])  # [B, P, out_features]
        outputs.append(out)

    outputs = torch.stack(outputs, dim=1)  # [B, N, P, out_features]
    print("outputs:", outputs.shape)
    print(outputs)

    # Final output
    y = torch.einsum("b n p d, b n p -> b d", outputs, combine_weights)
    print("y:", y.shape)
    print(y)

    return y


# === Run test ===
print("Input x:", x.shape)
print(x)

# Forward with inspection
output = forward_intercept(model, x)
