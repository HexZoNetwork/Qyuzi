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
        if self.queue.qsize() > 0:
            try:
                data = self.queue.get_nowait()
                if isinstance(data, tuple):
                    text, img_urls = data
                else:
                    text, img_urls = data, []
                
                tokens = encode("<|endoftext|>" + text)
                if len(tokens) > 100:
                   self.buffer.append((torch.tensor(tokens), img_urls))
            except Exception as e:
                pass
        
        while len(self.buffer) < 10:
             pad_token = getattr(config, 'PAD_TOKEN', 0)
             seq_len = config.MAX_SEQ
             x = torch.randint(0, config.VOCAB_SIZE, (seq_len,))
             self.buffer.append((x, []))
             break
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