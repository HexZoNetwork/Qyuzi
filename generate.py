import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn.functional as F
import argparse
from qyuzi.config import config
from qyuzi.model.transformer import QyuziUltimate
from qyuzi.data.tokenizer import encode, decode

@torch.inference_mode()
def generate(prompt: str, max_new=200, temperature=0.8, top_k=40):
    model = QyuziUltimate().to(config.DEVICE)
    model.eval()
    if model.load_checkpoint() is None:
        print("Warning: No checkpoint found, generating garbage.")
        
    ids = torch.tensor([encode(prompt)]).to(config.DEVICE)
    
    print(f"Generating for: '{prompt}'")
    
    logits, past_key_values = model(ids, think_steps=config.THINK_STEPS_INFER)
    
    for _ in range(max_new):
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('Inf')
        
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        
        ids = torch.cat([ids, next_id], dim=1)
        
        print(decode(next_id[0].tolist()), end='', flush=True)
        
        logits, past_key_values = model(next_id, think_steps=None, past_key_values=past_key_values)
    
    print("\n\nDone.")
    return decode(ids[0].tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="What Is Skibidi?")
    args = parser.parse_args()
    
    generate(args.prompt)
