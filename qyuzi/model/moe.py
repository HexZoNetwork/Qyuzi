import torch
import torch.nn as nn
import torch.nn.functional as F
from qyuzi.config import config

class ScalableMoE(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, num_experts: int = 8, top_k: int = 2, expert_capacity_ratio: float = 1.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.expert_capacity_ratio = expert_capacity_ratio
        
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.w1 = nn.Parameter(torch.randn(num_experts, hidden_dim, ffn_dim) * config.INIT_STD)
        self.w2 = nn.Parameter(torch.randn(num_experts, ffn_dim, hidden_dim) * config.INIT_STD)
        
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('expert_prob_sum', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.expert_counts.zero_()
            self.total_tokens = torch.tensor(0.0, device=x.device)
            self.expert_prob_sum.zero_()

        B, T, H = x.shape
        x_flat = x.view(-1, H)
        num_tokens = x_flat.size(0)

        router_logits = self.router(x_flat)
        if self.training:
             router_logits = router_logits + torch.randn_like(router_logits) * config.MOE_JITTER_NOISE
             
        router_probs = F.softmax(router_logits, dim=-1)
        k_probs, k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        k_probs = k_probs / k_probs.sum(dim=-1, keepdim=True)

        if self.training:
            with torch.no_grad():
                batch_counts = torch.bincount(k_indices.view(-1), minlength=self.num_experts)
                self.expert_counts += batch_counts.float()
                self.total_tokens += num_tokens
            self.expert_prob_sum += router_probs.sum(0)
        
        capacity = int(self.expert_capacity_ratio * num_tokens * self.top_k / self.num_experts) + 1
        
        indices_flat = k_indices.view(-1)
        sorted_indices, sort_map = torch.sort(indices_flat)
        
        counts = torch.bincount(sorted_indices, minlength=self.num_experts)
        
        output = torch.zeros_like(x_flat)
        
        start_indices = torch.cat([torch.tensor([0], device=x.device), torch.cumsum(counts, dim=0)[:-1]])
        
        for e in range(self.num_experts):
            count = counts[e].item()
            if count == 0:
                continue
                
            start = start_indices[e].item()
            end = start + count
            
            expert_token_indices = sort_map[start:end]
            
            if count > capacity:
                 perm = torch.randperm(count, device=x.device)[:capacity]
                 expert_token_indices = expert_token_indices[perm]
                 
            tokens = x_flat[expert_token_indices]
            
            h = F.silu(tokens @ self.w1[e]) 
            expert_out = h @ self.w2[e]
            
            output.index_add_(0, expert_token_indices, expert_out)

        weighted = output.view(num_tokens, self.top_k, H) * k_probs.unsqueeze(-1)
        final = weighted.sum(dim=1)

        return final.view(B, T, H) + x
    
    def load_balancing_loss(self) -> torch.Tensor:
        if self.total_tokens > 0:
            f_i = self.expert_counts / (self.total_tokens + 1e-6)
            P_i = self.expert_prob_sum / (self.total_tokens + 1e-6)
            var_f = torch.var(f_i)
            return self.num_experts * torch.sum(f_i * P_i) + 0.05 * var_f
        return torch.tensor(0.0, device=self.w1.device)
