import torch
from torch.utils.data import Dataset
from queue import Queue, Empty
import random
import tiktoken
from ..config import config

tokenizer = tiktoken.get_encoding("cl100k_base")

def encode(text): 
    return tokenizer.encode(text, allowed_special={'<|endoftext|>'})

def decode(ids): 
    return tokenizer.decode(ids)

class EndlessDataset(Dataset):
    def __init__(self, queue: Queue):
        self.queue = queue
        self.buffer = []

    def __len__(self): 
        # Infinite dataset concept
        return 1_000_000_000 

    def __getitem__(self, idx):
        # 1. Fill Buffer if low
        tries = 0
        while len(self.buffer) < 10:
            try:
                data = self.queue.get(timeout=2)
                if isinstance(data, tuple):
                    text, img_urls = data
                else:
                    text, img_urls = data, []
                
                tokens = encode("<|endoftext|>" + text)
                if len(tokens) > 100:
                    self.buffer.append((torch.tensor(tokens), img_urls))
                tries = 0
            except Empty:
                tries += 1
                if tries > 5:
                    if len(self.buffer) > 0:
                        break
                    # Emergency fallback data to prevent crash
                    fallback = "Science is the systematic study of the structure and behavior of the physical and natural world. " * 5
                    self.buffer.append((torch.tensor(encode(fallback)), []))
                    break
            except Exception:
                pass
        
        # 2. Sample from buffer
        chunk_data = random.choice(self.buffer)
        chunk, img_urls = chunk_data # Consistent unpacking
        
        # 3. Random crop
        if len(chunk) <= config.MAX_SEQ + 1:
             # Padding or short return (handled by collator usually, but here we just slice carefully)
             seq = chunk
        else:
            i = random.randint(0, len(chunk) - config.MAX_SEQ - 1)
            seq = chunk[i : i + config.MAX_SEQ + 1]
            
        x = seq[:-1]
        y = seq[1:]
        
        # Pad if short (simple zero padding for now)
        if len(x) < config.MAX_SEQ:
            pad = torch.zeros(config.MAX_SEQ - len(x), dtype=torch.long)
            x = torch.cat([x, pad])
            y = torch.cat([y, pad]) # -100 for ignore index would be better but keeping 0 for now as per legacy
            
        return x, y, torch.empty(0) # Returning empty for images for now
