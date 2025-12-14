import os
import sys
os.environ["QYUZI_STAGE"] = "f" 
os.environ["QYUZI_DATASET"] = "wikitext"
os.environ["QYUZI_REAL_DATA"] = "1"
os.environ["WANDB_MODE"] = "disabled"
from qyuzi.config import config
from qyuzi.data import tokenizer
class TestConfig(config.__class__):
    HIDDEN = 16 
    NUM_LAYERS = 2
    NUM_HEADS = 2
    FFN_DIM = 64
    VOCAB_SIZE = 1000
    MAX_SEQ = 64
    DEVICE = 'cpu'
    BATCH_SIZE = 2
    GRAD_ACCUM = 1
    LR = 1e-3
    USE_MOE = False
config.__class__ = TestConfig
config.ENABLE_SNN = False
config.ENABLE_VSA = False
config.ENABLE_DREAM = False
config.ENABLE_SELFMODEL = False
config.ENABLE_MULTIMODAL = False
config.USE_RECURRENT_THINKING = False
config.THINK_STEPS_TRAIN = 1
config.ENABLE_CHECKPOINTING = True
config.SAVE_INTERVAL = 10000
config.USE_REAL_DATASETS = True
tokenizer.HAS_TIKTOKEN = False
from train import train

if __name__ == "__main__":
    print(f"Config: {config.HIDDEN} hidden, {config.NUM_LAYERS} layers, {config.VOCAB_SIZE} vocab")
    print(f"Device: {config.DEVICE}")
    print(f"Using real dataset: {config.USE_REAL_DATASETS}")
    print(f"Dataset name: {config.DATASET_NAME}")
    
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    try:
        train(max_steps=50)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Done.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()