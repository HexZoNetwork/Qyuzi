
import os
import sys
import torch
import torch.nn.functional as F
os.environ["QYUZI_STAGE"] = "f" 
os.environ["QYUZI_DATASET"] = "synthetic"
os.environ["QYUZI_REAL_DATA"] = "0"

from qyuzi.config import config
from qyuzi.data import tokenizer
from qyuzi.model.transformer import QyuziUltimate
class TestConfig(config.__class__):
    HIDDEN = 16
    NUM_LAYERS = 2
    NUM_HEADS = 2
    FFN_DIM = 64
    VOCAB_SIZE = 258
    MAX_SEQ = 64
    USE_MOE = True
    NUM_EXPERTS = 4
    EXPERTS_ACTIVE = 1
    
# Apply Test Config
original_config_class = config.__class__
config.__class__ = TestConfig
config.ENABLE_SNN = False
config.ENABLE_VSA = False
config.ENABLE_DREAM = False
config.ENABLE_SELFMODEL = False
config.ENABLE_MULTIMODAL = False
config.USE_RECURRENT_THINKING = False
config.THINK_STEPS_TRAIN = 1
config.THINK_STEPS_INFER = 1
tokenizer.HAS_TIKTOKEN = False

def generate_text(prompt="Hello", max_new_tokens=20):
    model = QyuziUltimate().to(config.DEVICE)
    model.eval()
    
    # Load latest checkpoint if exists
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "qyuzi_latest.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=config.DEVICE)
        model.load_state_dict(ckpt['model_state'], strict=False)
    else:
        print("No checkpoint found, using random weights.")

    t = tokenizer.AutoTokenizer.get_instance()
    # SimpleTokenizer encode returns list of ints (bytes)
    input_ids = t.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long).to(config.DEVICE)
    
    print(f"\nPrompt: {prompt}")
    print("Generating...", end="", flush=True)
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs, _ = model(x, think_steps=1)
            if outputs.dim() == 4: outputs = outputs.squeeze(1)
            logits = outputs[:, -1, :] # (B, P)
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0) # Greedy
            
            try:
                x = torch.cat([x, next_token], dim=1)
            except Exception as ex:
                print(f"DEBUG: Cat Error. x.shape={x.shape}, next_token.shape={next_token.shape}")
                raise ex
            
            # Print token as we go (if printable)
            try:
                char = t.decode([next_token.item()])
                print(char, end="", flush=True)
            except:
                pass
                
    print("\n\nDone.")
    full_text = t.decode(x[0].tolist())
    print(f"Full Output: {full_text}")

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The "
    generate_text(prompt)
