import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn.functional as F
import torch.distributed as dist
import time
from datetime import datetime
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from qyuzi.config import config
from qyuzi.utils import seed_everything, setup_logging
from qyuzi.model.transformer import QyuziUltimate
from qyuzi.data.crawler import CognitiveCrawler
from qyuzi.data.dataset import EndlessDataset
from torch.utils.data import DataLoader, IterableDataset



def train(*args, **kwargs):
    # Device setup
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            config.DEVICE = f"cuda:{local_rank}"
        else:
            config.DEVICE = "cpu"
    else:
        config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    seed_everything(42)
    logger = setup_logging()
    
    # Model setup
    logger.info(f"Using device: {config.DEVICE}")
    model = QyuziUltimate(**kwargs).to(config.DEVICE)
    if hasattr(torch, 'compile') and config.DEVICE.startswith("cuda"):
        print("Compiling model...")
        model = torch.compile(model)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    
    # Mixed Precision Checks
    use_amp = config.DEVICE.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    
    checkpoint = model.load_checkpoint()
    start_step = 0
    if checkpoint:
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_step = checkpoint['step'] + 1
    try:
        import wandb
        wandb.init(
            project="qyuzi", 
            name=f"{config.VERSION}-{datetime.now().strftime('%Y%m%d-%H%M%S')}", 
            config=vars(config), 
            resume=True if start_step > 0 else "allow"
        )
        logger.info("WandB initialized")
    except ImportError:
        logger.warning("WandB not found")
        
    queue = Queue(maxsize=5000)
    
    executor = ThreadPoolExecutor(max_workers=10)
    if not config.USE_REAL_DATASETS:
        crawler = CognitiveCrawler(queue)
        executor.submit(crawler.run)
        crawler.start()
    if config.USE_REAL_DATASETS:
        try:
            from datasets import load_dataset
            if config.DATASET_NAME == 'wikitext':
                hf_ds = load_dataset(config.DATASET_NAME, 'wikitext-103-raw-v1', split='train', streaming=True)
            else:
                hf_ds = load_dataset(config.DATASET_NAME, split='train', streaming=True)
                
            def hf_data_generator():
                for item in hf_ds:
                    text = item.get('text', '')
                    if text: yield text
            hf_dataset = hf_data_generator()
            
            # Critical Fix: Feed HF data into the queue via thread
            def hf_feeder():
                for text in hf_dataset:
                    queue.put((text, []))
                    # Prevent queue overflow
                    while queue.qsize() > 4000:
                        time.sleep(0.1)
            executor.submit(hf_feeder)
            logger.info("Started HF Dataset Feeder thread.")
            
        except Exception as e:
            logger.warning(f"Failed to load HF dataset: {e}. Falling back to Crawler.")
            hf_dataset = None

    dataset = EndlessDataset(queue) 
    
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, num_workers=0, pin_memory=(config.DEVICE == "cuda"))

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    logger.info(f"Starting training from step {start_step}")
    
    step = start_step
    accum_steps = 0
    losses = []
    start_time = time.time()
    
    for batch in loader:
        x, y, images = batch
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        if step < config.WARMUP_STEPS:
            lr_mult = step / config.WARMUP_STEPS
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.LR * lr_mult
        else:
            progress = (step - config.WARMUP_STEPS) / config.TOTAL_STEPS
            cosine_lr = config.LR * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            for param_group in optimizer.param_groups:
                 param_group['lr'] = max(cosine_lr, config.LR * 0.1)

        device_type = 'cuda' if config.DEVICE.startswith("cuda") else 'cpu'
        
        # CPU autocast is often slow or requires bfloat16, disabling for simplicity/stability if CPU
        # Or you can use: torch.amp.autocast(device_type=device_type, enabled=use_amp)
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            output = model(x, think_steps=config.THINK_STEPS_TRAIN, images=images)
            msg = "Model output must be a tuple or tensor"
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            loss = F.cross_entropy(logits.view(-1, config.VOCAB_SIZE), y.view(-1), ignore_index=-100)
            if config.USE_MOE:
                moe_loss = model.get_moe_loss()
                loss = loss + config.MOE_LOAD_BALANCE_WEIGHT * moe_loss
            
            if config.ENABLE_DREAM and step % 10 == 0:
                if hasattr(model, 'dream') and model.dream:
                     model.dream.consolidate_async()
                pass

        if scaler:
            scaler.scale(loss / config.GRAD_ACCUM).backward()
            accum_steps += 1
            if accum_steps % config.GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accum_steps = 0
        else:
            (loss / config.GRAD_ACCUM).backward()
            accum_steps += 1
            if accum_steps % config.GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_steps = 0

        losses.append(loss.item())
        if step % 50 == 0:
            avg_loss = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - start_time
            steps_per_sec = (step - start_step + 1) / elapsed if elapsed > 0 else 0
            logger.info(f"[Step {step}] Loss: {avg_loss:.4f} | {steps_per_sec:.2f} step/s | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            if 'wandb' in locals():
                wandb.log({"loss": loss.item(), "avg_loss": avg_loss, "lr": optimizer.param_groups[0]['lr']}, step=step)

        if step % config.SAVE_INTERVAL == 0 and step > 0:
            model.save_checkpoint(step, loss.item(), optimizer)
        
        if kwargs.get('max_steps') and step >= kwargs['max_steps']:
            logger.info(f"Max steps {kwargs['max_steps']} reached. Stopping.")
            break

        step += 1

if __name__ == "__main__":
    train()
