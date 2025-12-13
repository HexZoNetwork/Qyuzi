import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import json
from datetime import datetime
from qyuzi.config import config
from qyuzi.model.layers import RMSNorm, SwiGLUMLP, Context32KScaling, RecurrentGate
from qyuzi.model.moe import ScalableMoE
from qyuzi.model.modules import (
    AdvancedSpikingNeuralNetwork, VectorSymbolicArchitecture, DreamConsolidationEngine,
    SelfModelingModule, RecursiveSelfImprovement, ConsciousWorkingMemory, CausalEngine
)

class UnifiedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = RMSNorm(config.HIDDEN)
        self.qkv = nn.Linear(config.HIDDEN, config.HIDDEN*3, bias=False)
        self.attn_scale = Context32KScaling(config.HIDDEN, config.NUM_HEADS, max_seq_len=config.MAX_SEQ)
        self.proj = nn.Linear(config.HIDDEN, config.HIDDEN, bias=False)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        
        self.ln2 = RMSNorm(config.HIDDEN)
        if config.USE_MOE:
            self.moe = ScalableMoE(config.HIDDEN, config.FFN_DIM, config.NUM_EXPERTS, config.EXPERTS_ACTIVE)
        else:
            self.mlp = SwiGLUMLP(config.HIDDEN, config.FFN_DIM)
    
    def forward(self, x, mask=None):
        if config.ENABLE_CHECKPOINTING and self.training and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mask, use_reentrant=False)
        return self._forward(x, mask)

    def _forward(self, x, mask=None):
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        attn_out = self.attn_scale(q, k, v, mask=mask)
        x = x + self.dropout(self.proj(attn_out))
        h = self.ln2(x)
        if hasattr(self, 'moe'):
            x = x + self.dropout(self.moe(h))
        else:
            x = x + self.dropout(self.mlp(h))
        return x

class QyuziUltimate(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(config.VOCAB_SIZE, config.HIDDEN)
        self.blocks = nn.ModuleList([
            UnifiedBlock() for _ in range(config.NUM_LAYERS)
        ])
        
        self.moe_layers = [block.moe for block in self.blocks if hasattr(block, 'moe')]
        self.register_buffer("causal_mask", torch.triu(torch.ones(config.MAX_SEQ, config.MAX_SEQ) * float('-inf'), diagonal=1))
        
        self.norm = RMSNorm(config.HIDDEN)
        self.lm_head = nn.Linear(config.HIDDEN, config.VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.embed.weight

        self.wm = ConsciousWorkingMemory(config.HIDDEN)
        self.causal = CausalEngine(config.HIDDEN)
        self.snn = AdvancedSpikingNeuralNetwork(config.HIDDEN, config.HIDDEN) if config.ENABLE_SNN else None
        self.vsa = VectorSymbolicArchitecture(dim=10000, seed_dim=config.HIDDEN) if config.ENABLE_VSA else None
        self.dream = DreamConsolidationEngine(config.HIDDEN) if config.ENABLE_DREAM else None
        self.self_model = SelfModelingModule(config.HIDDEN) if config.ENABLE_SELFMODEL else None
        self.self_improvement = RecursiveSelfImprovement(config.HIDDEN) if config.ENABLE_AUTONOMY else None
        if config.ENABLE_MULTIMODAL:
             try:
                 import open_clip
                 self.vision_encoder, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
                 for param in self.vision_encoder.parameters(): param.requires_grad = False
                 self.vision_proj = nn.Linear(512, config.HIDDEN)
             except ImportError:
                 print("OpenCLIP not found. Multimodal disabled.")
                 self.vision_encoder = None
        else:
             self.vision_encoder = None
             
        self.recurrent_gate = RecurrentGate(config.HIDDEN)
        self.think_norm = RMSNorm(config.HIDDEN)

    def get_moe_loss(self):
        if not self.moe_layers:
            return 0.0
        return sum(moe.load_balancing_loss() for moe in self.moe_layers) / len(self.moe_layers)

    def forward(self, idx, think_steps=None, images=None):
        if think_steps is None: 
            think_steps = config.THINK_STEPS_TRAIN
        x = self.embed(idx)
        if config.ENABLE_MULTIMODAL and images is not None and self.vision_encoder is not None:
             with torch.no_grad():
                 img_feat = self.vision_encoder.encode_image(images)
             img_emb = self.vision_proj(img_feat).unsqueeze(1)
             x = torch.cat([img_emb, x], dim=1)
        
        T = x.shape[1]
        active_mask = self.causal_mask[:T, :T]
        if active_mask.shape != (T, T):
             active_mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)

        hidden_prev = None
        tau = 3.0
        
        for step in range(think_steps):
            alpha = torch.exp(torch.tensor(-step / tau, device=x.device))
            for i, block in enumerate(self.blocks):
                if self.training and config.ENABLE_CHECKPOINTING and i % 4 == 0:
                     x = torch.utils.checkpoint.checkpoint(block, x, active_mask, use_reentrant=False)
                else:
                     x = block(x, mask=active_mask)

            if self.snn:
                 snn_out = self.snn(x)
                 gate = torch.sigmoid(self.lm_head(snn_out))
                 x = x * (1 + config.SNN_FEEDBACK_SCALE * gate * snn_out.mean(dim=-1, keepdim=True))
            
            x_recurrent = self.recurrent_gate(x.mean(dim=1), hidden_prev)
            hidden_prev = x_recurrent
            x = x + alpha * (x_recurrent.unsqueeze(1) * config.RECURRENT_RESIDUAL_SCALE)
            
            wm_out = self.wm(x.mean(1))
            x = x + wm_out
            
            if T > 3:
                c = x[:, :-3].mean(1)
                e = x[:, 3:].mean(1)
                prob = self.causal(c, e)[:, 1]
                x[:, 3:] += config.CAUSAL_BRANCH_SCALE * x[:, :-3] * prob.unsqueeze(-1).unsqueeze(-1)
            
            x = self.think_norm(x)
            
        if self.dream and self.training:
            importance = torch.rand(x.shape[0], T, device=x.device)
            self.dream.store_experience(x.detach(), importance, vsa=self.vsa)

        return self.lm_head(self.norm(x))

    def save_checkpoint(self, step, loss, optimizer=None):
        checkpoint = {
            'step': step, 'loss': loss, 'model_state': self.state_dict(),
            'config': vars(config), 'timestamp': datetime.now().isoformat(),
            'version': config.VERSION
        }
        if optimizer: checkpoint['optimizer_state'] = optimizer.state_dict()
        
        if not os.path.exists(config.CHECKPOINT_DIR):
            os.makedirs(config.CHECKPOINT_DIR)
            
        path = os.path.join(config.CHECKPOINT_DIR, f"qyuzi_{config.VERSION}_step{step}.pt")
        torch.save(checkpoint, path)
        latest_path = os.path.join(config.CHECKPOINT_DIR, "qyuzi_latest.pt")
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, path=None):
        if path is None: path = os.path.join(config.CHECKPOINT_DIR, "qyuzi_latest.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=config.DEVICE)
            self.load_state_dict(checkpoint['model_state'], strict=False)
            print(f"Loaded checkpoint from {path}")
            return checkpoint
        return None
