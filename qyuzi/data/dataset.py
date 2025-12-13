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
        return 1_000_000_000 

    def __getitem__(self, idx):
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
                    fallback = "Science is the systematic study of the structure and behavior of the physical and natural world. " * 5
                    self.buffer.append((torch.tensor(encode(fallback)), []))
                    break
            except Exception:
                pass
        chunk_data = random.choice(self.buffer)
        chunk, img_urls = chunk_data
        if len(chunk) <= config.MAX_SEQ + 1:
             seq = chunk
        else:
            i = random.randint(0, len(chunk) - config.MAX_SEQ - 1)
            seq = chunk[i : i + config.MAX_SEQ + 1]
            
        x = seq[:-1]
        y = seq[1:]
        if len(x) < config.MAX_SEQ:
            pad = torch.zeros(config.MAX_SEQ - len(x), dtype=torch.long)
            x = torch.cat([x, pad])
            y = torch.cat([y, pad])
            
        return x, y, torch.empty(0)