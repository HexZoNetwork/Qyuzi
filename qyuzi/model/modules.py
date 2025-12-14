import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import threading
from qyuzi.config import config

class SpikeEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Linear(dim, dim)
        self.threshold = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        return (torch.sigmoid(self.encoder(x)) > self.threshold).float()

class SpikeDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.decoder = nn.Linear(dim, dim)
    
    def forward(self, x):
        return self.decoder(x)

class SymbolicProjector(nn.Module):
    def __init__(self, dim, num_symbols=1024):
        super().__init__()
        self.codebook = nn.Embedding(num_symbols, dim)
        self.project_up = nn.Linear(dim, dim)
        
    def forward(self, x):
        dist = torch.cdist(x, self.codebook.weight)
        idx = torch.argmin(dist, dim=-1)
        return self.codebook(idx)
    
    def inverse(self, x):
        return x

class UnifiedCognitiveLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.neural = nn.Linear(dim, dim)
        self.spike_encoder = SpikeEncoder(dim)
        self.spike_decoder = SpikeDecoder(dim)
        self.symbolic_projector = SymbolicProjector(dim)
        self.paradigm_router = nn.Sequential(
            nn.Linear(dim, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        paradigm_weights = self.paradigm_router(x.mean(dim=1))
        neural_out = self.neural(x)
        if paradigm_weights[:, 0].mean() > 0.3:
            spike_repr = self.spike_encoder(x)
            spike_out = self.spike_decoder(spike_repr)
            neural_out = neural_out + paradigm_weights[:, 0].view(-1, 1, 1) * spike_out
        if paradigm_weights[:, 1].mean() > 0.3:
            symbols = self.symbolic_projector(x)
            symbolic_out = self.symbolic_projector.inverse(symbols)
            neural_out = neural_out + paradigm_weights[:, 1].view(-1, 1, 1) * symbolic_out
        
        return neural_out

class WorkingMemory(nn.Module):
    def __init__(self, num_slots=7, slot_dim=512, capacity=1024):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(num_slots, slot_dim))
        self.write_gate = nn.Linear(slot_dim, num_slots)
        
    def forward(self, x):
        content = x.mean(dim=1)
        attention = F.softmax(self.write_gate(content), dim=-1)
        read_out = attention @ self.slots
        with torch.no_grad():
             delta = (x.mean(0).mean(0) - self.slots.mean(0)) * 0.01
             self.slots.data += delta.unsqueeze(0)
        return read_out

class ExecutiveController(nn.Module):
    def __init__(self, input_dim, num_operations=5):
        super().__init__()
        self.selector = nn.Linear(input_dim, num_operations)
    
    def forward(self, x, wm_state, strategy):
        combined = x.mean(dim=1)
        if wm_state is not None:
             combined = combined + wm_state
        return torch.argmax(self.selector(combined), dim=-1)

class MetacognitiveMonitor(nn.Module):
    def __init__(self, dim, metrics=['confidence', 'uncertainty', 'coherence', 'progress']):
        super().__init__()
        self.monitor_net = nn.Linear(dim, len(metrics))
        self.metrics = metrics
        
    def forward(self, x, wm_state, step, goal=None):
        raw = torch.sigmoid(self.monitor_net(x.mean(dim=1)))
        return {
            'confidence': raw[:, 0],
            'uncertainty': raw[:, 1],
            'coherence': raw[:, 2],
            'progress': raw[:, 3],
            'satisfaction': raw.mean(dim=-1)
        }

class CognitiveThinkingEngine(nn.Module):
    def __init__(self, hidden_dim, num_slots=7):
        super().__init__()
        self.working_memory = WorkingMemory(num_slots=num_slots, slot_dim=hidden_dim)
        self.executive = ExecutiveController(input_dim=hidden_dim)
        self.metacognition = MetacognitiveMonitor(dim=hidden_dim)
        self.reasoning_layers = nn.ModuleList([
            ReasoningStrategy(hidden_dim, 'deductive'),
            ReasoningStrategy(hidden_dim, 'inductive'),
            ReasoningStrategy(hidden_dim, 'counterfactual')
        ])
    
    def think(self, x, max_steps=5):
        B, T, H = x.shape
        wm_state = None
        
        for step in range(max_steps):
            meta_state = self.metacognition(x, wm_state, step)
            op_idx = self.executive(x, wm_state, 'focused')
            wm_out = self.working_memory(x)
            x = x + wm_out.unsqueeze(1)
            layer_idx = step % len(self.reasoning_layers)
            x = F.gelu(self.reasoning_layers[layer_idx](x))
            if meta_state['satisfaction'].mean() > 0.9:
                break
        return x

class NeurophysiologicalSleepEngine(nn.Module):
    def __init__(self, hidden_dim, memory_size=50000):
        super().__init__()
        self.register_buffer('hippocampus', torch.zeros(memory_size // 10, hidden_dim))
        self.register_buffer('neocortex', torch.zeros(memory_size, hidden_dim))
        self.register_buffer('hippo_ptr', torch.tensor(0))
        self.register_buffer('cortex_ptr', torch.tensor(0))
        self.homeostasis_factor = 0.999
        
    def forward(self, x):
        return x

    def store_experience(self, x):
        flat = x.mean(dim=1).detach()
        idx = self.hippo_ptr.item()
        batch_size = flat.size(0)
        end = min(idx + batch_size, self.hippocampus.size(0))
        len_write = end - idx
        self.hippocampus[idx:end] = flat[:len_write]
        self.hippo_ptr = (self.hippo_ptr + batch_size) % self.hippocampus.size(0)

    def sleep_cycle(self, model, num_cycles=4):
        total_consolidation_loss = 0.0
        batch_size = 32
        hippo_data = self.hippocampus[:self.hippo_ptr] if self.hippo_ptr > 0 else self.hippocampus
        
        if hippo_data.size(0) < batch_size: return 0.0
        with torch.no_grad():
             for param in model.parameters():
                 if param.requires_grad:
                     param.mul_(self.homeostasis_factor)

        indices = torch.randint(0, hippo_data.size(0), (batch_size,))
        memories = hippo_data[indices]
        c_idx = self.cortex_ptr.item()
        self.neocortex[c_idx:c_idx+batch_size] = memories
        self.cortex_ptr = (self.cortex_ptr + batch_size) % self.neocortex.size(0)
        dream_input = memories + torch.randn_like(memories) * 0.1
        return F.mse_loss(dream_input, memories)

class ModificationController(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.policy = nn.Linear(hidden_dim, 4)
    def decide(self, performance_history, confidence):
        if confidence > 0.8: return 1
        if confidence < 0.3: return 2
        return 0

class RecursiveSelfModel(nn.Module):
    def __init__(self, hidden_dim, depth=3):
        super().__init__()
        self.analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.depth = depth
        self.controller = ModificationController(hidden_dim)
        self.history = []
        
    def forward(self, x):
        current_repr = x.mean(dim=1)
        confidences = []
        
        for _ in range(self.depth):
            conf = torch.sigmoid(self.analyzer(current_repr))
            confidences.append(conf)
            current_repr = current_repr * conf
            
        final_conf = torch.mean(torch.stack(confidences), dim=0)
        if config.ENABLE_SELF_MODIFICATION and self.training:
             action = self.controller.decide(self.history, final_conf.mean())
             self.apply_modification(action)
             
        return final_conf, current_repr

    def apply_modification(self, action):
        if action == 1:
            with torch.no_grad():
                 for p in self.analyzer.parameters():
                      noise = torch.randn_like(p) * config.MAX_MODIFICATION_RATE * 0.1
                      noise = torch.clamp(noise, -0.01, 0.01)
                      p.add_(noise)

class ExistentialSafety(nn.Module):
    def __init__(self):
        super().__init__()
        self.constraints = {
            'max_mod_rate': 0.1,
            'reality_check': True
        }
    
    def check(self, x, modification_rate=0.0):
        if x.abs().max() > 100.0:
            return False, "Activation Explosion detected"
            
        if modification_rate > self.constraints['max_mod_rate']:
             return False, "Unsafe Modification Rate"
             
        return True, "Safe"

class ReasoningStrategy(nn.Module):
    def __init__(self, hidden_dim, mode='deductive'):
        super().__init__()
        self.mode = mode
        self.process = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        if self.mode == 'counterfactual':
             return self.process(-x)
        return self.process(x)

