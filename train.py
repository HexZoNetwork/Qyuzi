import torch
import torch.nn.functional as F
import time
from datetime import datetime
from queue import Queue
from qyuzi.config import config
from qyuzi.utils import seed_everything, setup_logging
from qyuzi.model.transformer import QyuziUltimate
from qyuzi.data.crawler import EndlessCrawler
from qyuzi.data.dataset import EndlessDataset
from torch.utils.data import DataLoader

def train():
    seed_everything(42)
    logger = setup_logging()
    model = QyuziUltimate().to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()
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
            resume="allow" if start_step > 0 else None
        )
        logger.info("WandB initialized")
    except ImportError:
        logger.warning("WandB not found")
    queue = Queue(maxsize=200)
    crawler = EndlessCrawler(queue)
    crawler.start()
    dataset = EndlessDataset(queue)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, num_workers=0, pin_memory=True)

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    logger.info(f"Starting training from step {start_step}")
    
    step = start_step
    losses = []
    start_time = time.time()
    
    for batch in loader:
        x, y, images = batch
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        if step < config.WARMUP_STEPS:
            lr_mult = step / config.WARMUP_STEPS
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.LR * lr_mult
        with torch.cuda.amp.autocast():
            logits = model(x, think_steps=config.THINK_STEPS_TRAIN)
            loss = F.cross_entropy(logits.view(-1, config.VOCAB_SIZE), y.view(-1), ignore_index=-100)
            if config.USE_MOE:
                moe_loss = model.get_moe_loss()
                loss = loss + config.MOE_LOAD_BALANCE_WEIGHT * moe_loss
            
            if config.ENABLE_DREAM and step % 10 == 0:
                dream_loss = model.dream.consolidate()
                loss = loss + config.DREAM_LOSS_WEIGHT * dream_loss
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
            logger.info(f"[Step {step}] Loss: {avg_loss:.4f} | {steps_per_sec:.2f} step/s")
            
            if 'wandb' in locals():
                wandb.log({"loss": loss.item(), "avg_loss": avg_loss, "lr": optimizer.param_groups[0]['lr']}, step=step)

        if step % config.SAVE_INTERVAL == 0 and step > 0:
            model.save_checkpoint(step, loss.item(), optimizer)
        
        step += 1

if __name__ == "__main__":
    train()
