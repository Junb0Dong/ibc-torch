import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Dummy data: N=100 samples, o: 2D observation, a: 2D action (m=2)
N = 100
dim_o = 2
m = 2  # action dimensions
o = torch.randn(N, dim_o)
a = torch.randn(N, m)  # expert actions

# Action bounds for negative samples
a_min = -2.0
a_max = 2.0
N_neg = 32  # negative samples per positive

# Autoregressive Energy Models: m sub-models E^j(o, a[:j])
class SubEnergyModel(nn.Module):
    def __init__(self, input_dim_o, input_dim_a):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim_o + input_dim_a, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # scalar energy
        )
    
    def forward(self, o, a_partial):
        # o: [N, dim_o]
        # a_partial: [N, j] for pos, [N, N_neg, j] for neg
        if a_partial.dim() == 2:
            # Positive
            o_rep = o
            input = torch.cat([o_rep, a_partial], dim=-1)
            energies = self.net(input).squeeze(-1)
        else:
            # Negative: [N, N_neg, j]
            o_rep = o.unsqueeze(1).repeat(1, a_partial.shape[1], 1)
            input = torch.cat([o_rep, a_partial], dim=-1)
            input_flat = input.view(-1, input.shape[-1])
            energies_flat = self.net(input_flat).squeeze(-1)
            energies = energies_flat.view(o.shape[0], -1)
        return energies

# Initialize m sub-models
models = nn.ModuleList([SubEnergyModel(dim_o, j) for j in range(1, m+1)])
print("models: ", models)

# Optimizer: Adam on all parameters
optimizer = optim.Adam([p for model in models for p in model.parameters()], lr=1e-3)

# InfoNCE Loss for dimension j
def info_nce_loss_j(model_j, o_batch, a_pos_batch, a_neg_batch_j):
    # a_pos_batch: [N, j], a_neg_batch_j: [N, N_neg, j]
    pos_energy = model_j(o_batch, a_pos_batch)  # [N]
    neg_energies = model_j(o_batch, a_neg_batch_j)  # [N, N_neg]
    # For stability
    pos_energy = torch.clamp(pos_energy, -10, 10)
    neg_energies = torch.clamp(neg_energies, -10, 10)
    neg_sum = torch.logsumexp(-neg_energies, dim=1)
    denom = torch.exp(-pos_energy) + torch.exp(neg_sum)
    loss = -torch.log(torch.exp(-pos_energy) / denom + 1e-8).mean()
    return loss

# Training loop: 100 epochs
num_epochs = 1000
losses = []

# 对应论文Niters
for epoch in range(num_epochs): 
    optimizer.zero_grad()
    total_loss = 0.0
    
    # For each dimension j=1 to m
    for j in range(1, m+1):
        # Positive: a_pos[:j] [N, j]
        a_pos_j = a[:, :j]
        
        # Negative: [N, N_neg, j], uniform random
        a_neg_j = torch.empty(N, N_neg, j).uniform_(a_min, a_max)
        
        # Compute loss for j
        loss_j = info_nce_loss_j(models[j-1], o, a_pos_j, a_neg_j)
        total_loss += loss_j
    
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_([p for model in models for p in model.parameters()], 1.0)
    optimizer.step()
    losses.append(total_loss.item())
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Total Loss = {total_loss.item():.4f}")

# Final losses
print("Final average loss:", np.mean(losses[-10:]))

# Test: Compute average energy on positive vs negative
with torch.no_grad():
    avg_pos_energy = 0.0
    avg_neg_energy = 0.0
    for j in range(1, m+1):
        a_pos_j = a[:, :j]
        a_neg_j = torch.empty(N, N_neg, j).uniform_(a_min, a_max)
        pos_e = models[j-1](o, a_pos_j).mean().item()
        neg_e = models[j-1](o, a_neg_j).mean().item()
        avg_pos_energy += pos_e / m
        avg_neg_energy += neg_e / m
    print(f"Avg Positive Energy: {avg_pos_energy:.4f}")
    print(f"Avg Negative Energy: {avg_neg_energy:.4f}")