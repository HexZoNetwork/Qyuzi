import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from qyuzi.config import config

# --- SNN Components ---
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input, threshold)
        return (input >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, threshold = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Rectangular surrogate gradient
        spike_pseudo_grad = (torch.abs(input - threshold) < 0.5).float()
        return grad_input * spike_pseudo_grad, -grad_input * spike_pseudo_grad

class SpikingNeuronLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, tau_mem=20.0, threshold=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = math.exp(-1.0 / tau_mem)
        self.base_threshold = threshold
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.threshold_adapt = nn.Parameter(torch.ones(hidden_dim) * threshold)
        
    def forward(self, x, mem=None):
        B, T, _ = x.shape
        if mem is None:
            mem = torch.zeros(B, self.hidden_dim, device=x.device)
        spikes = []
        for t in range(T):
            mem = self.alpha * mem + self.fc(x[:, t])
            # Adaptive threshold: avoid NaN if batch size is 1 by setting unbiased=False or checking dim
            if B > 1:
                mem_std = mem.std(dim=-1, keepdim=True)
            else:
                mem_std = torch.zeros_like(mem[:, 0:1])
                
            threshold = self.threshold_adapt.unsqueeze(0) + 0.1 * mem_std
            spike = SurrogateSpike.apply(mem, threshold)
            mem = mem * (1 - spike)
            spikes.append(spike)
        return torch.stack(spikes, dim=1), mem

class AdvancedSpikingNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            SpikingNeuronLayer(input_dim if i == 0 else hidden_dim, hidden_dim, threshold=1.0)
            for i in range(num_layers)
        ])
        self.inhibition = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        
    def forward(self, x):
        h = x
        mem = None
        for layer in self.layers:
            h, mem = layer(h, mem)
            h = h * (1 - self.inhibition.unsqueeze(0).unsqueeze(0))
        return h

# --- VSA Components ---
class VectorSymbolicArchitecture(nn.Module):
    def __init__(self, dim=10000, seed_dim=512, num_symbols=1000):
        super().__init__()
        self.dim = dim
        self.seed_dim = seed_dim
        self.codebook = nn.Embedding(num_symbols, dim)
        self.project_up = nn.Linear(seed_dim, dim)
        self.project_down = nn.Linear(dim, seed_dim)
        self.resonator = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        
    def bind(self, a, b):
        a_fft = torch.fft.fft(a, dim=-1)
        b_fft = torch.fft.fft(b, dim=-1)
        return torch.fft.ifft(a_fft * b_fft, dim=-1).real
    
    def cleanup(self, x):
        refined = self.resonator(x)
        return F.normalize(refined, dim=-1)
    
    def forward(self, concept_a, concept_b=None, operation='bind'):
        hv_a = self.project_up(concept_a)
        if concept_b is not None:
            hv_b = self.project_up(concept_b)
            if operation == 'bind':
                result = self.bind(hv_a, hv_b)
            elif operation == 'bundle':
                result = F.normalize(hv_a + hv_b, dim=-1)
            else:
                result = self.bind(hv_a, hv_b)
        else:
            result = hv_a
             
        result = self.cleanup(result)
        return self.project_down(result)

    def retrieve(self, query, threshold=0.7):
        if query.shape[-1] == self.seed_dim:
            query_hv = self.project_up(query)
        else:
            query_hv = query
        codebook_norm = F.normalize(self.codebook.weight, dim=-1)
        query_norm = F.normalize(query_hv, dim=-1)
        sim = torch.matmul(query_norm, codebook_norm.T)
        return sim
import threading

class DreamConsolidationEngine(nn.Module):
    def __init__(self, hidden_dim, memory_size=50000, compression_ratio=4, num_dream_cycles=10):
        super().__init__()
        self.memory_size = memory_size
        self.compressed_dim = hidden_dim // compression_ratio
        self.num_dream_cycles = num_dream_cycles
        # Offload to CPU
        self.episodic_memory = torch.zeros(memory_size, hidden_dim, device='cpu')
        self.memory_importance = torch.zeros(memory_size, device='cpu')
        self.write_ptr = 0
        
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.compressed_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.compressed_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.world_model = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        
    def store_experience(self, hidden_states, importance=None, vsa=None):
        B, T, H = hidden_states.shape
        states_flat = hidden_states.view(-1, H)
        if importance is None:
            importance = torch.ones(states_flat.size(0), device=states_flat.device)
        else:
            importance = importance.view(-1).clamp(0.0, 5.0)
        if vsa is not None:
             states_flat = vsa(states_flat, operation='cleanup')
        states_cpu = states_flat.detach().cpu()
        imp_cpu = importance.detach().cpu()
        num_items = states_cpu.size(0)
        indices = [(self.write_ptr + i) % self.memory_size for i in range(num_items)]
        for i, idx in enumerate(indices):
             self.episodic_memory[idx] = states_cpu[i]
             self.memory_importance[idx] = imp_cpu[i]
        self.write_ptr += num_items
    
    def dream(self, num_samples=100, device='cuda'):
        if self.write_ptr == 0: return torch.tensor(0.0, device=device)
        
        probs = F.softmax(self.memory_importance, dim=0)
        idx = torch.multinomial(probs, num_samples, replacement=True)
        samples = self.episodic_memory[idx].to(device)
        
        encoded = self.encoder(samples)
        mu, logvar = encoded.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        reconstructed = self.decoder(z)
        
        world_input = reconstructed.unsqueeze(0)
        world_output, _ = self.world_model(world_input)
        
        recon_loss = F.mse_loss(reconstructed, samples, reduction='none').mean(dim=-1)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        world_loss = F.mse_loss(world_output.squeeze(0), samples, reduction='none').mean(dim=-1)
        total_error = recon_loss + 0.1 * world_loss + 0.001 * kl_loss
        with torch.no_grad():
             err_cpu = total_error.detach().cpu()
             self.memory_importance[idx] = 0.9 * self.memory_importance[idx] + 0.1 * err_cpu
        return total_error.mean()
    
    def consolidate(self):
        total_loss = 0.0
        device = next(self.parameters()).device
        for _ in range(self.num_dream_cycles):
            loss = self.dream(num_samples=128, device=device)
            total_loss += loss
        return total_loss / self.num_dream_cycles

    def consolidate_async(self):
        def _job():
            with torch.no_grad():
                self.consolidate()
        threading.Thread(target=_job).start()

class ConsciousWorkingMemory(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(9, hidden_dim) * config.INIT_STD)
        self.gate = nn.Linear(hidden_dim, 9)

    def forward(self, x):
        scores = self.gate(x.mean(dim=1, keepdim=True))
        attn = F.softmax(scores, -1) @ self.slots
        mean_x = x.mean(dim=(0,1))
        with torch.no_grad():
            self.slots.data.copy_(0.99 * self.slots.data + 0.01 * mean_x.unsqueeze(0).repeat(9,1))
        return attn.unsqueeze(1)
class CausalEngine(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim*2, 256),
            nn.GELU(),
            nn.Linear(256, 3)
        )
    def forward(self, a, b):
        return F.softmax(self.net(torch.cat([a,b], -1)), -1)

class SelfModelingModule(nn.Module):
    def __init__(self, hidden_dim, num_capabilities=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.probe = nn.Linear(hidden_dim, num_capabilities)
        
    def forward(self, x):
         return self.probe(x)

class RecursiveSelfImprovement(nn.Module):
    def __init__(self, hidden_dim, num_iterations=5):
        super().__init__()
        self.net = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x):
        return {'improvements': torch.tanh(self.net(x))}
