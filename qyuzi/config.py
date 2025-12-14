import os
import torch

class StageConfig:
    STAGES = {
        "f": {
            "hidden": 1024,
            "layers": 30,
            "heads": 16,
            "ffn": 4096,
            "params": "670M",
            "use_moe": False
        },
        "sc": {
            "hidden": 1280,
            "layers": 36,
            "heads": 20,
            "ffn": 6144,
            "params": "1.1B",
            "use_moe": False
        },
        "th": {
            "hidden": 1280,
            "layers": 36,
            "heads": 20,
            "ffn": 6144,
            "params": "1.5B",
            "use_moe": True,
            "num_experts": 8,
            "experts_active": 2
        },
        "fh": {
            "hidden": 1536,
            "layers": 40,
            "heads": 24,
            "ffn": 8192,
            "params": "3B",
            "use_moe": True,
            "num_experts": 16,
            "experts_active": 2
        },
        "fih": {
            "hidden": 2048,
            "layers": 48,
            "heads": 32,
            "ffn": 10240,
            "params": "8B",
            "use_moe": True,
            "num_experts": 32,
            "experts_active": 2
        }
    }

class Config:
    ACTIVE_STAGE = os.getenv("QYUZI_STAGE", "f")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if ACTIVE_STAGE not in StageConfig.STAGES:
        raise ValueError(f"Stage {ACTIVE_STAGE} not valid. Choose: {list(StageConfig.STAGES.keys())}")
    
    stage_cfg = StageConfig.STAGES[ACTIVE_STAGE]
    VERSION = ACTIVE_STAGE
    try:
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")
        VOCAB_SIZE = tokenizer.n_vocab
        # enc.eot_token is not always available dependent on version, usually 100257
        EOT_TOKEN = getattr(tokenizer, 'eot_token', 100257) 
        PAD_TOKEN = 100258 # Synthetic ID for pad if using tiktoken
    except ImportError:
        VOCAB_SIZE = 258 
        EOT_TOKEN = 256
        PAD_TOKEN = 257
    CRAWLER_TOPICS = ["science", "history", "philosophy", "technology", "mathematics", "biology", "physics", "ai", "quantum mechanics", "neuroscience", "space exploration"]
    USE_SYNTHETIC_FALLBACK = True

    HIDDEN = stage_cfg["hidden"]
    NUM_LAYERS = stage_cfg["layers"]
    NUM_HEADS = stage_cfg["heads"]
    FFN_DIM = stage_cfg["ffn"]
    USE_MOE = stage_cfg["use_moe"]
    NUM_EXPERTS = stage_cfg.get("num_experts", 8)
    EXPERTS_ACTIVE = stage_cfg.get("experts_active", 2)
    MAX_SEQ = 8192
    ROPE_THETA = 10000.0
    ROPE_SCALING_FACTOR = 1.0
    USE_RECURRENT_THINKING = os.getenv("QYUZI_RECURRENT", "0") == "1" 
    THINK_STEPS_TRAIN = 1 if USE_RECURRENT_THINKING else 3
    THINK_STEPS_INFER = 1 if USE_RECURRENT_THINKING else 5
    BATCH_SIZE = 8 if HIDDEN > 1024 else 12
    GRAD_ACCUM = 4 if HIDDEN > 1024 else 3
    LR = 1.5e-4 if HIDDEN > 1024 else 2e-4
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 1000
    USE_REAL_DATASETS = os.getenv("QYUZI_REAL_DATA", "1") == "1"
    DATASET_NAME = os.getenv("QYUZI_DATASET", "HuggingFaceFW/fineweb-edu")
    DEDUP_CACHE_SIZE = 100000
    SAVE_INTERVAL = 1000
    CHECKPOINT_DIR = "qyuzi_checkpoints"
    ENABLE_CHECKPOINTING = os.getenv("QYUZI_CHECKPOINTING", "1") == "1"
    ENABLE_SNN = os.getenv("QYUZI_SNN", "0") == "1"
    ENABLE_VSA = os.getenv("QYUZI_VSA", "0") == "1"
    ENABLE_DREAM = os.getenv("QYUZI_DREAM", "0") == "1"
    ENABLE_SELFMODEL = os.getenv("QYUZI_SELFMODEL", "0") == "1"
    ENABLE_MULTIMODAL = os.getenv("QYUZI_MULTIMODAL", "0") == "1"
    ENABLE_AUDIO = os.getenv("QYUZI_AUDIO", "0") == "1"
    ENABLE_VIDEO = os.getenv("QYUZI_VIDEO", "0") == "1"
    ENABLE_AUTONOMY = os.getenv("QYUZI_AUTONOMY", "0") == "1"
    MOE_JITTER_NOISE = 0.01
    MOE_LOAD_BALANCE_WEIGHT = 0.01
    DREAM_LOSS_WEIGHT = 0.001
    SELF_IMPROVEMENT_RATE = 0.1
    SNN_FEEDBACK_SCALE = 0.1
    VSA_CONTEXT_SCALE = 0.05
    RECURRENT_RESIDUAL_SCALE = 0.1
    WM_SCALE = 1.0
    CAUSAL_BRANCH_SCALE = 0.15
    INIT_STD = 0.02
    TOTAL_STEPS = 1000000
    DROPOUT_RATE = 0.1
    
    # Cognitive Architecture Config
    COGNITIVE_MEMORY_SLOTS = 7
    COGNITIVE_SLEEP_CYCLES = 4
    SELF_MODEL_DEPTH = 3
    CURIOSITY_THRESHOLD = 0.6
    
    # Advanced Phases (3 & 4)
    ENABLE_SELF_MODIFICATION = True
    MAX_MODIFICATION_RATE = 0.01
    ENABLE_COUNTERFACTUALS = True
    ENABLE_EXISTENTIAL_SAFETY = True

config = Config()
