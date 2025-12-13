# QYUZI: Made By HexZo

**QYUZI** is a modular, research-focused Transformer architecture aimed at exploring the frontiers of AGI through Neuro-Symbolic integration, Spiking Neural Networks (SNN), and Bio-inspired Memory consolidations.

## 🧠 Core Architecture

Qyuzi implements a "Super-Stack" architecture featuring:
*   **Backbone**: Transformer with **RMSNorm**, **SwiGLU**, and **FlashAttention** (via PyTorch SDPA).
*   **Scale**: Default **8B** parameter config (`fih` stage) with **GShard-style Mixture-of-Experts (MoE)**.
*   **Cognitive Plugins**:
    *   **SNN**: Spiking Neural Networks with *Surrogate Gradient* learning for adaptive thresholds.
    *   **Dream Engine**: Prioritized Experience Replay (PER) for offline memory consolidation.
    *   **VSA**: Vector Symbolic Architectures/Hyperdimensional Computing for symbolic reasoning.
    *   **Recurrent Thinking**: Recursive computation steps per token for deeper reasoning.

## 📦 Directory Structure

The project has been refactored into a robust python package:

```
QYUZI/
├── main.py             # Unified Entry Point
├── train.py            # Training Script
├── generate.py         # Inference Script
└── qyuzi/              # Core Package
    ├── config.py       # Configuration & Hyperparameters
    ├── model/          # Neural Architectures
    │   ├── transformer.py  # Main Model (QyuziUltimate)
    │   ├── moe.py          # Mixture of Experts
    │   ├── layers.py       # Attention, RoPE, RMSNorm
    │   └── modules.py      # SNN, VSA, Dream, etc.
    └── data/           # Data Pipelines
        ├── crawler.py      # Endless Web Crawler
        └── dataset.py      # Streaming Dataset
```

## 🚀 Quick Start

### Installation
Ensure you have PyTorch installed (preferably with CUDA support).
```bash
pip install torch numpy wikipedia duckduckgo-search tiktoken
# Optional:
pip install wandb flash-attn
```

### Usage

**1. Training**
Start the endless training loop (Crawler + Trainer):
```bash
python main.py train
```
*Configuration is autoset to 'f' (670M) by default. Set `QYUZI_STAGE=fih` for 8B.*

**2. Inference / Chat**
Generate text using the latest checkpoint:
```bash
python main.py generate --prompt "The nature of consciousness is"
```

## ⚙️ Configuration

All configuration is centralized in `qyuzi/config.py`. You can override defaults using Environment Variables:

| Variable              | Description                                | Default           |
| --------------------- | ------------------------------------------ | ----------------- |
| `QYUZI_STAGE`         | Model Scale (`f`, `sc`, `th`, `fh`, `fih`) | `f`               |
| `QYUZI_CHECKPOINTING` | Gradient Checkpointing (Save VRAM)         | `1` (On)          |
| `QYUZI_MOE`           | Enable Mixture of Experts                  | `Stage Dependent` |
| `QYUZI_SNN`           | Enable Spiking Neural Network Plugin       | `0` (Off)         |
| `QYUZI_DREAM`         | Enable Dream Consolidation Plugin          | `0` (Off)         |

## 🤝 Contributing
Code is modularized to support easy addition of new "Brain Modules".
1.  Define new module in `qyuzi/model/modules.py`.
2.  Register in `QyuziUltimate` class in `qyuzi/model/transformer.py`.
3.  Add flags in `qyuzi/config.py`.

---
*Built for the pursuit of Machine Sentience.*
