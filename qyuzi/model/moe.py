import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import config

class ScalableMoE(nn.Module):
    """
    Mixture of Experts Layer with GShard-style Load Balancing and Top-K Routing.
    """
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
            self.total_tokens.zero_()
            self.expert_prob_sum.zero_()

        B, T, H = x.shape
        x_flat = x.view(-1, H)
        num_tokens = x_flat.size(0)

        router_logits = self.router(x_flat)
        # Jitter noise for stability
        if self.training:
             router_logits = router_logits + torch.randn_like(router_logits) * config.MOE_JITTER_NOISE
             
        router_probs = F.softmax(router_logits, dim=-1)
        k_probs, k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Normalize probabilities for the selected top-k
        k_probs = k_probs / k_probs.sum(dim=-1, keepdim=True)

        if self.training:
            with torch.no_grad():
                batch_counts = torch.bincount(k_indices.view(-1), minlength=self.num_experts)
                self.expert_counts += batch_counts.float()
                self.total_tokens += num_tokens
            # GShard prob sum tracking
            self.expert_prob_sum += router_probs.sum(0)

        # Capacity enforcing
        capacity = int(self.expert_capacity_ratio * num_tokens * self.top_k / self.num_experts)
        capacity = max(4, capacity)
        indices_flat = k_indices.view(-1)
        sorted_indices, sort_map = torch.sort(indices_flat)
        counts = torch.bincount(sorted_indices, minlength=self.num_experts)
        counts_clamped = torch.clamp(counts, max=capacity)

        output = torch.zeros_like(x_flat)
        
        # Expert execution loop (Optimized for single-device simulation)
        # In a real distributed setting, this would involve all-to-all dispatch
        start = 0
        for e in range(self.num_experts):
            count = counts[e].item()
            if count == 0:
                continue
            clamped = counts_clamped[e].item()

            # Gather tokens for expert e
            tokens = x_flat[sort_map[start:start + count]]
            
            # FFN computation
            # Note: w1[e] is [H, FFN], tokens is [N, H] -> [N, FFN]
            h = F.silu(tokens @ self.w1[e]) 
            expert_out = h @ self.w2[e]

            if count > clamped:
                # Drop overflow tokens
                expert_out = expert_out[:clamped]

            # Scatter results back
            positions = sort_map[start:start + clamped]
            output[positions] += expert_out
            start += count
        
        # Re-weight and sum top-k contributions
        weighted = output.view(num_tokens, self.top_k, H) * k_probs.unsqueeze(-1)
        final = weighted.sum(dim=1)

        return final.view(B, T, H) + x
    
    def load_balancing_loss(self) -> torch.Tensor:
        if self.total_tokens > 0:
            f_i = self.expert_counts / (self.total_tokens + 1e-6)
            P_i = self.expert_prob_sum / (self.total_tokens + 1e-6)
            return self.num_experts * torch.sum(f_i * P_i)
        return torch.tensor(0.0, device=self.w1.device)
