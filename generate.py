import torch
import torch.nn.functional as F
import argparse
from qyuzi.config import config
from qyuzi.model.transformer import QyuziUltimate
from qyuzi.data.dataset import encode, decode

@torch.inference_mode()
def generate(prompt: str, max_new=200, temperature=0.8, top_k=40):
    model = QyuziUltimate().to(config.DEVICE)
    model.eval()
    if model.load_checkpoint() is None:
        print("Warning: No checkpoint found, generating garbage.")
        
    ids = torch.tensor([encode(prompt)]).to(config.DEVICE)
    
    print(f"Generating for: '{prompt}'")
    for _ in range(max_new):
        logits = model(ids, think_steps=config.THINK_STEPS_INFER)
        logits = logits[:, -1, :] / temperature
        
        # Top-K
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('Inf')
        
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
        
        print(decode(next_id[0].tolist()), end='', flush=True)
    
    print("\n\nDone.")
    return decode(ids[0].tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The future of AGI is")
    args = parser.parse_args()
    
    generate(args.prompt)
