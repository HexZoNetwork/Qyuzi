import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import json
from datetime import datetime
from qyuzi.config import config
from .layers import RMSNorm, SwiGLUMLP, Context32KScaling, RecurrentGate
from .moe import ScalableMoE
from .modules import (
    AdvancedSpikingNeuralNetwork, VectorSymbolicArchitecture, DreamConsolidationEngine,
    SelfModelingModule, RecursiveSelfImprovement, ConsciousWorkingMemory, CausalEngine
)

class NativeViT(nn.Module):
    def __init__(self, patch_size=16, hidden_dim=512, num_layers=5):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        
    def encode_image(self, images):
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        B, L, D = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.blocks(x)
        return self.norm(x[:, 0])

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
    
    def forward(self, x, mask=None, past_key_value=None):
        if config.ENABLE_CHECKPOINTING and self.training and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mask, past_key_value, use_reentrant=False)
        return self._forward(x, mask, past_key_value)

    def _forward(self, x, mask=None, past_key_value=None):
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        
        if past_key_value is not None:
             k_past, v_past = past_key_value
             k = torch.cat([k_past, k], dim=1)
             v = torch.cat([v_past, v], dim=1)
        
        current_kv = (k, v)
             
        attn_out = self.attn_scale(q, k, v, mask=mask)
        x = x + self.dropout(self.proj(attn_out))
        h = self.ln2(x)
        if hasattr(self, 'moe'):
            x = x + self.dropout(self.moe(h))
        else:
            x = x + self.dropout(self.mlp(h))
            
        return x, current_kv

class QyuziUltimate(nn.Module):
    def __init__(self, **kwargs):
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
                  self.vision_encoder, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
                  for param in self.vision_encoder.parameters(): param.requires_grad = False
                  self.vision_proj = nn.Linear(512, config.HIDDEN)
             except ImportError:
                  print("OpenCLIP not found. Falling back to NativeViT.")
                  self.vision_encoder = NativeViT(hidden_dim=512)
                  self.vision_proj = nn.Linear(512, config.HIDDEN)
        else:
             self.vision_encoder = None
             
        self.recurrent_gate = RecurrentGate(config.HIDDEN)
        self.think_norm = RMSNorm(config.HIDDEN)

    def get_moe_loss(self):
        if not self.moe_layers:
            return 0.0
        return sum(moe.load_balancing_loss() for moe in self.moe_layers) / len(self.moe_layers)

    def forward(self, idx, think_steps=None, images=None, past_key_values=None):
        if think_steps is None: 
            think_steps = config.THINK_STEPS_TRAIN
        
        x = self.embed(idx)
        
        if config.ENABLE_MULTIMODAL and images is not None and self.vision_encoder is not None:
             if not torch.is_tensor(images):
                 images = torch.tensor(images)
             if past_key_values is None:
                 with torch.no_grad():
                     img_feat = self.vision_encoder.encode_image(images)
                 img_emb = self.vision_proj(img_feat).unsqueeze(1)
                 x = torch.cat([img_emb, x], dim=1)
        
        T = x.shape[1]
        
        if past_key_values is not None:
             past_len = past_key_values[0][0].shape[1]
             total_len = past_len + T
             active_mask = self.causal_mask[past_len:total_len, :total_len]
        else:
             active_mask = self.causal_mask[:T, :T]

        if active_mask.shape[0] != T:
             active_mask = torch.triu(torch.ones(T, T + (past_len if past_key_values else 0), device=x.device) * float('-inf'), diagonal=1)

        hidden_prev = None
        tau = 3.0
        
        new_past_key_values = []
        
        meta_hidden = None

        if past_key_values is not None:
            think_steps = 1

        for step in range(think_steps):
            alpha = torch.exp(torch.tensor(-step / tau, device=x.device))
            
            if self.self_improvement:
                improvement_context, meta_hidden = self.self_improvement(x, meta_hidden)
                x = x + 0.1 * improvement_context

            current_layer_kv = []
            
            for i, block in enumerate(self.blocks):
                
                block_past = past_key_values[i] if past_key_values else None
                
                if self.training and config.ENABLE_CHECKPOINTING and i % 4 == 0:
                     x, kv = torch.utils.checkpoint.checkpoint(block, x, active_mask, block_past, use_reentrant=False)
                else:
                     x, kv = block(x, mask=active_mask, past_key_value=block_past)
                
                current_layer_kv.append(kv)

            if self.self_model:
                capabilities, confidence = self.self_model(x)
                adaptive_scale = confidence.view(-1, 1, 1)
            else:
                adaptive_scale = 1.0

            if self.snn:
                 snn_out = self.snn(x)
                 gate = torch.sigmoid(self.lm_head(snn_out))
                 x = x * (1 + config.SNN_FEEDBACK_SCALE * adaptive_scale * gate * snn_out.mean(dim=-1, keepdim=True))
            
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
            
            new_past_key_values = current_layer_kv
            
        if self.dream and self.training:
            importance = torch.rand(x.shape[0], T, device=x.device)
            self.dream.store_experience(x.detach(), importance, vsa=self.vsa)

        return self.lm_head(self.norm(x)), new_past_key_values

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
