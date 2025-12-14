import torch
from torch.utils.data import Dataset
from queue import Queue, Empty
import random
from .tokenizer import encode, decode
from qyuzi.config import config

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
                    fallback_tokens = [0] * (config.MAX_SEQ + 1)
                    self.buffer.append((torch.tensor(fallback_tokens, dtype=torch.long), []))
                    break
            except Exception as e:
                print(f"Dataset Internal Error: {e}")
                tries += 1
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