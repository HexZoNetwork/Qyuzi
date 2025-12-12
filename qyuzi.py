import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import requests
import time
import random
import os
from datetime import datetime
import threading
from queue import Queue
import wikipedia
import duckduckgo_search
from duckduckgo_search import DDGS
import json
import math
import hashlib
from typing import Optional, Tuple, List
from collections import deque
import numpy as np
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("meh using pytorch native..")

try:
    import datasets
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False
    print("hell no hf datasets")

class StageConfig:
    STAGES = {
        "f": {
            "hidden": 1024,
            "layers": 30,
            "heads": 16,
            "ffn": 4096,
            "params": "670M",
            "use_moe": False
        },
        "sc": {
            "hidden": 1280,
            "layers": 36,
            "heads": 20,
            "ffn": 6144,
            "params": "1.1B",
            "use_moe": False
        },
        "th": {
            "hidden": 1280,
            "layers": 36,
            "heads": 20,
            "ffn": 6144,
            "params": "1.5B",
            "use_moe": True,
            "num_experts": 8,
            "experts_active": 2
        },
        "fh": {
            "hidden": 1536,
            "layers": 40,
            "heads": 24,
            "ffn": 8192,
            "params": "3B",
            "use_moe": True,
            "num_experts": 16,
            "experts_active": 2
        },
        "fih": {
            "hidden": 2048,
            "layers": 48,
            "heads": 32,
            "ffn": 10240,
            "params": "8B",
            "use_moe": True,
            "num_experts": 32,
            "experts_active": 2
        }
    }

class Config:
    ACTIVE_STAGE = os.getenv("QYUZI_STAGE", "f")
    
    stage_cfg = StageConfig.STAGES[ACTIVE_STAGE]
    VERSION = ACTIVE_STAGE
    
    VOCAB_SIZE = 131072
    HIDDEN = stage_cfg["hidden"]
    NUM_LAYERS = stage_cfg["layers"]
    NUM_HEADS = stage_cfg["heads"]
    FFN_DIM = stage_cfg["ffn"]
    USE_MOE = stage_cfg["use_moe"]
    NUM_EXPERTS = stage_cfg.get("num_experts", 8)
    EXPERTS_ACTIVE = stage_cfg.get("experts_active", 2)
    
    MAX_SEQ = 2048
    ROPE_THETA = 10000.0
    ROPE_SCALING_FACTOR = 1.0
    USE_RECURRENT_THINKING = os.getenv("QYUZI_RECURRENT", "0") == "1" 
    THINK_STEPS_TRAIN = 1 if USE_RECURRENT_THINKING else 3
    THINK_STEPS_INFER = 1 if USE_RECURRENT_THINKING else 5
    BATCH_SIZE = 8 if HIDDEN > 1024 else 12
    GRAD_ACCUM = 4 if HIDDEN > 1024 else 3
    LR = 1.5e-4 if HIDDEN > 1024 else 2e-4
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 1000
    
    USE_REAL_DATASETS = os.getenv("QYUZI_REAL_DATA", "1") == "1"
    DATASET_NAME = os.getenv("QYUZI_DATASET", "HuggingFaceFW/fineweb-edu")
    DEDUP_CACHE_SIZE = 100000
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    SAVE_INTERVAL = 1000
    CHECKPOINT_DIR = "qyuzi_checkpoints"
    
    ENABLE_SNN = os.getenv("QYUZI_SNN", "0") == "1"
    ENABLE_VSA = os.getenv("QYUZI_VSA", "0") == "1"
    ENABLE_DREAM = os.getenv("QYUZI_DREAM", "0") == "1"
    ENABLE_SELFMODEL = os.getenv("QYUZI_SELFMODEL", "0") == "1"

config = Config()

os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")

def encode(text): return tokenizer.encode(text, allowed_special={'<|endoftext|>'})
def decode(ids): return tokenizer.decode(ids)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=32768, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_seq_len)
    
    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class ConsciousWorkingMemory(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(9, hidden_dim) * 0.02)
        self.gate = nn.Linear(hidden_dim, 9)

    def forward(self, x):
        scores = self.gate(x.mean(dim=1, keepdim=True))
        attn = F.softmax(scores, -1) @ self.slots
        mean_x = x.mean(dim=(0,1))
        self.slots.data = 0.99 * self.slots.data + 0.01 * mean_x.unsqueeze(0).repeat(9,1)
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

class MoELayer(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_experts=8, top_k=2, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.capacity_factor = capacity_factor
        
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        
        self.w1 = nn.Parameter(torch.randn(num_experts, hidden_dim, ffn_dim) * 0.02)
        self.w2 = nn.Parameter(torch.randn(num_experts, ffn_dim, hidden_dim) * 0.02)
        
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        
    def forward(self, x):
        B, T, H = x.shape
        x_flat = x.view(-1, H)
        N = x_flat.shape[0]
        
        router_logits = self.router(x_flat)
        if self.training:
            router_logits = router_logits + torch.randn_like(router_logits) * 0.01
        
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        output = torch.zeros_like(x_flat)
        
        capacity = max(1, int(self.capacity_factor * N / self.num_experts * self.top_k))
        
        for k in range(self.top_k):
            expert_ids = top_k_indices[:, k]
            weights = top_k_probs[:, k].unsqueeze(-1)
            
            for expert_id in range(self.num_experts):
                mask = (expert_ids == expert_id)
                if not mask.any():
                    continue
                
                # Get tokens for this expert
                mask_indices = torch.nonzero(mask).squeeze(-1)
                
                # Apply capacity limit
                if mask_indices.shape[0] > capacity:
                    mask_indices = mask_indices[:capacity]
                
                if mask_indices.shape[0] == 0:
                    continue
                
                tokens = x_flat[mask_indices]
                
                h = torch.einsum('nh,hf->nf', tokens, self.w1[expert_id])
                h = F.gelu(h)
                h = torch.einsum('nf,fh->nh', h, self.w2[expert_id])
                
                output[mask_indices] += weights[mask_indices] * h
                self.expert_counts[expert_id] += mask_indices.shape[0]
        
        return output.view(B, T, H)
    
    def load_balancing_loss(self):
        counts_norm = self.expert_counts / (self.expert_counts.sum() + 1e-10)
        target = torch.ones_like(counts_norm) / self.num_experts
        return F.mse_loss(counts_norm, target)

class QyuziLamborghini(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(config.VOCAB_SIZE, config.HIDDEN)
        self.pos = nn.Embedding(config.MAX_SEQ, config.HIDDEN)
        
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            config.HIDDEN, 
            config.NUM_HEADS, 
            dim_feedforward=config.FFN_DIM,
            batch_first=True, 
            activation='gelu', 
            dropout=0.1
        ) for _ in range(config.NUM_LAYERS)])
        
        self.norm = nn.LayerNorm(config.HIDDEN)
        self.lm_head = nn.Linear(config.HIDDEN, config.VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.embed.weight

        self.wm = ConsciousWorkingMemory(config.HIDDEN)
        self.causal = CausalEngine(config.HIDDEN)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"🏎️  LAMBORGHINI ({config.VERSION}) ready — {total_params:,} params ({total_params/1e6:.1f}M)")
    
    def save_checkpoint(self, step, loss, optimizer=None):
        checkpoint = {
            'step': step,
            'loss': loss,
            'model_state': self.state_dict(),
            'config': vars(config),
            'timestamp': datetime.now().isoformat(),
            'version': config.VERSION
        }
        if optimizer:
            checkpoint['optimizer_state'] = optimizer.state_dict()
        
        path = os.path.join(config.CHECKPOINT_DIR, f"qyuzi_{config.VERSION}_step{step}.pt")
        torch.save(checkpoint, path)
        
        latest_path = os.path.join(config.CHECKPOINT_DIR, "qyuzi_latest.pt")
        torch.save(checkpoint, latest_path)
        
        meta_path = os.path.join(config.CHECKPOINT_DIR, "metadata.json")
        metadata = {
            'latest_step': step,
            'latest_loss': loss,
            'version': config.VERSION,
            'timestamp': checkpoint['timestamp'],
            'total_params': sum(p.numel() for p in self.parameters())
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"The Another One: step {step}, loss {loss:.4f}")
    
    def load_checkpoint(self, path=None):
        if path is None:
            path = os.path.join(config.CHECKPOINT_DIR, "qyuzi_latest.pt")
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=config.DEVICE)
            self.load_state_dict(checkpoint['model_state'])
            print(f"Anadar load")
            return checkpoint
        else:
            print(f"Fresh?")
            return None

    def forward(self, idx, think_steps=None):
        if think_steps is None: think_steps = config.THINK_STEPS_TRAIN
        B, T = idx.shape
        x = self.embed(idx) + self.pos(torch.arange(T, device=idx.device))

        for step in range(think_steps):
            for block in self.blocks:
                x = block(x)
            wm_out = self.wm(x.mean(1))
            x = x + wm_out.unsqueeze(1)

            if T > 3:
                c = x[:, :-3].mean(1)
                e = x[:, 3:].mean(1)
                prob = self.causal(c, e)[:, 1]  # prob a→b
                x[:, 3:] += 0.08 * x[:, :-3] * prob.unsqueeze(-1).unsqueeze(-1)

        return self.lm_head(self.norm(x))

class QyuziUltimate(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(config.VOCAB_SIZE, config.HIDDEN)
        self.pos = nn.Embedding(config.MAX_SEQ, config.HIDDEN)
        
        if config.USE_MOE:
            self.blocks = nn.ModuleList([
                self._build_moe_block() for _ in range(config.NUM_LAYERS)
            ])
            self.moe_layers = [block.moe for block in self.blocks if hasattr(block, 'moe')]
        else:
            self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(
                config.HIDDEN,
                config.NUM_HEADS,
                dim_feedforward=config.FFN_DIM,
                batch_first=True,
                activation='gelu',
                dropout=0.1
            ) for _ in range(config.NUM_LAYERS)])
            self.moe_layers = []
        
        self.norm = nn.LayerNorm(config.HIDDEN)
        self.lm_head = nn.Linear(config.HIDDEN, config.VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.embed.weight

        self.wm = ConsciousWorkingMemory(config.HIDDEN)
        self.causal = CausalEngine(config.HIDDEN)

        total_params = sum(p.numel() for p in self.parameters())
        active_params = self._estimate_active_params() if config.USE_MOE else total_params
        
        print(f"🚀 QYUZI ({config.VERSION}) ready")
        print(f"   Total params: {total_params:,} ({total_params/1e6:.1f}M)")
        if config.USE_MOE:
            print(f"   Active params: {active_params:,} ({active_params/1e6:.1f}M)")
            print(f"   MoE: {config.NUM_EXPERTS}, top-{config.EXPERTS_ACTIVE} routing")

    def _build_moe_block(self):
        class MoEBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.MultiheadAttention(config.HIDDEN, config.NUM_HEADS, batch_first=True)
                self.moe = MoELayer(config.HIDDEN, config.FFN_DIM, config.NUM_EXPERTS, config.EXPERTS_ACTIVE)
                self.norm1 = nn.LayerNorm(config.HIDDEN)
                self.norm2 = nn.LayerNorm(config.HIDDEN)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x):
                attn_out, _ = self.self_attn(x, x, x)
                x = self.norm1(x + self.dropout(attn_out))
                moe_out = self.moe(x)
                x = self.norm2(x + self.dropout(moe_out))
                return x
        
        return MoEBlock()
    
    def _estimate_active_params(self):
        base = sum(p.numel() for n, p in self.named_parameters() if 'experts' not in n)
        expert_params = sum(p.numel() for n, p in self.named_parameters() if 'experts' in n)
        active_expert_params = expert_params * (config.EXPERTS_ACTIVE / config.NUM_EXPERTS)
        return int(base + active_expert_params)
    
    def save_checkpoint(self, step, loss, optimizer=None):
        checkpoint = {
            'step': step,
            'loss': loss,
            'model_state': self.state_dict(),
            'config': vars(config),
            'timestamp': datetime.now().isoformat(),
            'version': config.VERSION
        }
        if optimizer:
            checkpoint['optimizer_state'] = optimizer.state_dict()
        
        path = os.path.join(config.CHECKPOINT_DIR, f"qyuzi_{config.VERSION}_step{step}.pt")
        torch.save(checkpoint, path)
        
        latest_path = os.path.join(config.CHECKPOINT_DIR, "qyuzi_latest.pt")
        torch.save(checkpoint, latest_path)
        
        meta_path = os.path.join(config.CHECKPOINT_DIR, "metadata.json")
        metadata = {
            'latest_step': step,
            'latest_loss': loss,
            'version': config.VERSION,
            'timestamp': checkpoint['timestamp'],
            'total_params': sum(p.numel() for p in self.parameters()),
            'use_moe': config.USE_MOE
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Anadar one step {step}, loss {loss:.4f}")
    
    def load_checkpoint(self, path=None):
        if path is None:
            path = os.path.join(config.CHECKPOINT_DIR, "qyuzi_latest.pt")
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=config.DEVICE)
            self.load_state_dict(checkpoint['model_state'])
            print(f"anadar load")
            return checkpoint
        else:
            print(f"fresh?")
            return None

    def forward(self, idx, think_steps=None):
        if think_steps is None: think_steps = config.THINK_STEPS_TRAIN
        B, T = idx.shape
        x = self.embed(idx) + self.pos(torch.arange(T, device=idx.device))

        for step in range(think_steps):
            for block in self.blocks:
                x = block(x)
            wm_out = self.wm(x.mean(1))
            x = x + wm_out.unsqueeze(1)

            if T > 3:
                c = x[:, :-3].mean(1)
                e = x[:, 3:].mean(1)
                prob = self.causal(c, e)[:, 1]
                x[:, 3:] += 0.08 * x[:, :-3] * prob.unsqueeze(-1).unsqueeze(-1)

        return self.lm_head(self.norm(x))
    
    def get_moe_loss(self):
        if not self.moe_layers:
            return 0.0
        return sum(moe.load_balancing_loss() for moe in self.moe_layers) / len(self.moe_layers)

class EndlessCrawler(threading.Thread):
    def __init__(self, queue):
        super().__init__(daemon=True)
        self.queue = queue
        self.topics = ["science", "history", "philosophy", "technology", "mathematics", "biology", 
                      "physics", "psychology", "artificial intelligence", "consciousness", "causality", "AI", "earth", "space", "gnome","sybau"]

    def run(self):
        wikipedia.set_lang("en")
        while True:
            try:
                if random.random() < 0.6:
                    topic = random.choice(self.topics)
                    page = wikipedia.page(wikipedia.search(topic, results=1)[0])
                    text = page.content
                else:
                    with DDGS() as ddgs:
                        query = random.choice(self.topics) + " explained"
                        results = [r for r in ddgs.text(query, max_results=5)]
                        text = " ".join([r['body'] for r in results if r['body']])

                if len(text) > 500:
                    self.queue.put(text)
                    print(f"[{datetime.now()}] Crawled {len(text):,} chars — {topic}")
                time.sleep(random.uniform(3, 12))
            except Exception as e:
                time.sleep(10)

class EndlessDataset(Dataset):
    def __init__(self, queue):
        self.queue = queue
        self.buffer = []

    def __len__(self): return 1_000_000_000_000 #Lol

    def __getitem__(self, idx):
        while len(self.buffer) < 10:
            try:
                text = self.queue.get(timeout=5)
                tokens = encode("<|endoftext|>" + text)
                if len(tokens) > 100:
                    self.buffer.append(torch.tensor(tokens))
            except:
                pass

        chunk = random.choice(self.buffer)
        i = random.randint(0, len(chunk)-config.MAX_SEQ-10)
        seq = chunk[i:i+config.MAX_SEQ+10]
        return seq[:-1], seq[1:]

def endless_think_training():
    model = QyuziUltimate().to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    checkpoint = model.load_checkpoint()
    start_step = 0
    if checkpoint:
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_step = checkpoint['step'] + 1

    queue = Queue(maxsize=200)
    crawler = EndlessCrawler(queue)
    crawler.start()
    dataset = EndlessDataset(queue)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, num_workers=4, pin_memory=True)

    step = start_step
    start_time = time.time()
    losses = []
    
    print(f"Booting Lol")
    print(f"Starting from step {start_step}")

    for x, y in loader:
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        if step < config.WARMUP_STEPS:
            lr_mult = step / config.WARMUP_STEPS
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.LR * lr_mult

        with torch.cuda.amp.autocast():
            logits = model(x, think_steps=config.THINK_STEPS_TRAIN)
            lm_loss = F.cross_entropy(logits.view(-1, config.VOCAB_SIZE), y.view(-1), ignore_index=-100)
            
            if config.USE_MOE:
                moe_loss = model.get_moe_loss()
                loss = lm_loss + 0.01 * moe_loss
            else:
                loss = lm_loss

        scaler.scale(loss / config.GRAD_ACCUM).backward()

        if (step + 1) % config.GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        losses.append(loss.item())
        
        if step % 50 == 0:
            avg_loss = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - start_time
            steps_per_sec = (step - start_step + 1) / elapsed if elapsed > 0 else 0
            mode_str = f"MoE-{config.NUM_EXPERTS}x{config.EXPERTS_ACTIVE}" if config.USE_MOE else "Dense"
            print(f"[Step {step}] Loss: {avg_loss:.4f} | {steps_per_sec:.2f} steps/s | {mode_str} | WM+Causal")

        if step % config.SAVE_INTERVAL == 0 and step > 0:
            model.save_checkpoint(step, loss.item(), optimizer)

        if step % 500 == 0 and step > 0:
            model.eval()
            with torch.no_grad():
                prompt = encode("what is the meaning of gasoline")
                ids = torch.tensor([prompt]).to(config.DEVICE)
                for _ in range(100):
                    logits = model(ids, think_steps=config.THINK_STEPS_INFER)
                    next_id = torch.multinomial(F.softmax(logits[:,-1]/0.8, -1), 1)
                    ids = torch.cat([ids, next_id], -1)
                print(f"My Brain Got Trigered AHhh Gojo: {decode(ids[0].tolist())}")
            model.train()

        step += 1

    print(f"🚀 {config.VERSION} never dies...")

@torch.inference_mode()
def generate(prompt: str, max_new=200, think_steps=8, temperature=0.8):
    model = QyuziUltimate().to(config.DEVICE)
    model.eval()
    model.load_checkpoint()
    ids = torch.tensor([encode(prompt)]).to(config.DEVICE)
    for _ in range(max_new):
        logits = model(ids, think_steps=think_steps)
        next_id = torch.multinomial(F.softmax(logits[:,-1]/temperature, -1), 1)
        ids = torch.cat([ids, next_id], -1)
    return decode(ids[0].tolist())

if __name__ == "__main__":
    endless_think_training()