# 🏎️ Qyuzi — Ferrari Lol

<div align="center">

```
  ██████╗ ██╗   ██╗██╗   ██╗███████╗██╗
 ██╔═══██╗╚██╗ ██╔╝██║   ██║╚══███╔╝██║
 ██║   ██║ ╚████╔╝ ██║   ██║  ███╔╝ ██║
 ██║▄▄ ██║  ╚██╔╝  ██║   ██║ ███╔╝  ██║
 ╚██████╔╝   ██║   ╚██████╔╝███████╗██║
  ╚══▀▀═╝    ╚═╝    ╚═════╝ ╚══════╝╚═╝
```

**A Scalable Small Language Model with Advanced Cognitive Architecture**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status](https://img.shields.io/badge/Status-Active_Development-yellow.svg)](https://github.com)

</div>

---

## 🚀 Overview

**Qyuzi** is a cutting-edge small language model (SSLM) designed to push the boundaries of efficient AGI. Built from scratch with PyTorch, Qyuzi implements state-of-the-art transformer architecture enhanced with advanced cognitive modules including:

- 🧠 **Conscious Working Memory** — 9-slot attention-based memory system
- ⚡ **Mixture of Experts (MoE)** — Dynamic expert routing for efficient scaling
- 🔄 **Rotary Position Embedding (RoPE)** — Extended context window support (up to 32K tokens)
- 🎯 **Causal Reasoning Engine** — Explicit causal inference mechanisms
- 🌌 **Multi-Stage Architecture** — Scalable from 670M to 8B parameters

Unlike typical language models, Qyuzi integrates neuromorphic computing concepts, vector-symbolic architectures, and dream-based memory consolidation to achieve human-like reasoning with minimal parameters.

---

## 📊 Model Stages

Qyuzi features a progressive scaling architecture:

| Stage           | Parameters | Layers | Hidden | FFN   | Experts  | Context | Status        |
| --------------- | ---------- | ------ | ------ | ----- | -------- | ------- | ------------- |
| **F** (Ferrari) | 670M       | 30     | 1024   | 4096  | Dense    | 16K     | ✅ Active      |
| **FH**          | 1.5B       | 36     | 1280   | 5120  | 8×2 MoE  | 16K     | 🚧 Development |
| **SEC**         | 3.8B       | 42     | 1536   | 7680  | 16×2 MoE | 32K     | 📋 Planned     |
| **FIH**         | 8B         | 48     | 2048   | 10240 | 32×2 MoE | 32K     | 📋 Planned     |

*Active parameters in MoE variants are ~40% of total due to sparse expert activation*

---

## ✨ Key Features

### 🏗️ Core Architecture
- **Transformer-based** with custom optimizations
- **Flash Attention** support for memory efficiency
- **Gradient checkpointing** for large-scale training
- **Mixed precision** (FP16/BF16) training
- **Rotary embeddings** for extended context

### 🧬 Advanced Modules
- **Conscious Working Memory** — Explicit storage slots for multi-step reasoning
- **Causal Engine** — Predicts causal relationships (cause/effect/spurious)
- **MoE Layers** — Up to 32 experts with top-k routing
- **Load Balancing** — Automatic expert utilization optimization
- **Auxiliary Loss** — Prevents expert collapse

### 🔬 Experimental Features
- **Spiking Neural Networks (SNN)** — Energy-efficient neuromorphic computing
- **Vector-Symbolic Architecture (VSA)** — Hyperdimensional reasoning
- **Dream Engine** — Nightly memory consolidation and replay
- **Self-Modeling** — Meta-cognitive awareness of own predictions
- **Quantum Swarm** — Distributed inference across multiple nodes

### 🛠️ Training & Inference
- **Continuous learning** from web-scale data
- **Dynamic thinking** — Adjustable reasoning steps (1-16+)
- **Temperature-based sampling** — Controllable creativity
- **Checkpoint management** — Auto-save with versioning
- **Multi-modal** (planned) — Vision and audio understanding

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.7+ (for GPU acceleration)
- 16GB+ RAM (32GB+ recommended for large stages)

### Setup

```bash
git clone https://github.com/HexZoNetwork/Qyuzi.git
cd Qyuzi

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tiktoken requests datasets flash-attn --no-build-isolation
 #optional
pip install snnTorch hyperdimensional
```

### Quick Start

```bash
python main.py --mode train --stage f

python main.py --mode generate --prompt "What is consciousness?" --tokens 200 --steps 8

python main.py --mode eval --stage f

python main.py --mode swarm --nodes 1000 --topology hierarchical
```

---

## 🎮 Usage

### Training

```bash
python main.py --mode train --stage f

python main.py --mode train --stage fh --snn --vsa --dream --selfmodel

python main.py --mode train --stage sec --multimodal
```

### Generation

```bash
python main.py --mode generate --prompt "Explain quantum mechanics" --tokens 500

python main.py --mode generate \
  --prompt "Solve the Riemann hypothesis" \
  --tokens 2000 \
  --steps 16 \
  --temp 0.7 \
  --stage fh

python main.py --mode generate \
  --prompt "Write a sci-fi story about AI" \
  --tokens 1000 \
  --temp 1.2 \
  --steps 4
```

### Swarm Deployment

```bash
python main.py --mode swarm --nodes 1000 --topology hierarchical

python main.py --mode swarm --nodes 500 --topology mesh

python main.py --mode swarm --nodes 2000 --topology ring
```

---

## 🧩 Architecture Deep Dive

### Conscious Working Memory

Qyuzi implements a 9-slot attention-based working memory inspired by cognitive science research:

```python
slots = [slot_1, slot_2, ..., slot_9]
gate_scores = softmax(linear(hidden_state))
working_memory = sum(gate_scores * slots)
```

### Mixture of Experts (MoE)

Dynamic expert routing reduces computation while scaling parameters:

```
Top-K Router → Expert Selection → Parallel Processing → Weighted Combination
           ↓                                                  ↑
    Load Balancing Loss ←────────────────────────────────────┘
```

- **Capacity Factor**: 1.25× for overflow handling
- **Expert Dropout**: Prevents over-specialization
- **Auxiliary Loss**: Ensures balanced expert usage

### Causal Reasoning Engine

Predicts causal relationships between events:

```
Event A + Event B → Neural Network → [CAUSE, EFFECT, SPURIOUS]
```

Enables counterfactual reasoning and intervention planning.

---

## 🗺️ Roadmap

### Phase 1: Foundation ✅
- [x] Core transformer architecture
- [x] 670M dense model (Ferrari)
- [x] RoPE positional encoding
- [x] Basic training pipeline

### Phase 2: Scaling 🚧
- [x] MoE implementation
- [ ] 1.5B MoE model (8 experts)
- [ ] 3.8B MoE model (16 experts)
- [ ] Context scaling to 32K tokens

### Phase 3: Advanced Cognition 📋
- [ ] Spiking Neural Network co-processor
- [ ] Vector-Symbolic Architecture integration
- [ ] Dream-based memory consolidation
- [ ] Self-modeling module

### Phase 4: Multi-Modal 📋
- [ ] Vision encoder (256×256 → 1024×1024)
- [ ] Video understanding
- [ ] Audio input/output (Whisper + VALL-E)
- [ ] Multi-modal reasoning

### Phase 5: Autonomy 🌟
- [ ] Recursive self-improvement
- [ ] Neuro-symbolic theorem proving (Lean 4)
- [ ] Robotic embodiment (ROS2)
- [ ] Chemical computing substrate

---

## 🔧 Configuration

Set model stage and features via environment variables:

```bash
export QYUZI_STAGE=fh           # f, fh, sec, fih
export QYUZI_SNN=1              # Enable spiking neurons
export QYUZI_VSA=1              # Enable vector-symbolic arch
export QYUZI_DREAM=1            # Enable dream engine
export QYUZI_SELFMODEL=1        # Enable self-modeling
```

Or use CLI flags:

```bash
python main.py --stage fh --snn --vsa --dream --selfmodel
```

---

## 📈 Performance

*Benchmarks:*

| Stage          | Training Speed | Inference (bs=1) | Memory Usage |
| -------------- | -------------- | ---------------- | ------------ |
| F (670M)       | ~2.3K tok/s    | ~45 tok/s        | 9.5 GB       |
| FH (1.5B MoE)  | ~1.8K tok/s    | ~35 tok/s        | 14.2 GB      |
| SEC (3.8B MoE) | ~950 tok/s     | ~22 tok/s        | 20.8 GB      |

*Prediction I Dont Have Gpu Lol*

---

## 🤝 Contributing

Qyuzi is an active research project. Contributions are welcome!

### Areas of Interest
- 🧠 Novel reasoning mechanisms
- ⚡ Training optimizations
- 🌐 Multi-modal architectures
- 🔬 Neuromorphic computing
- 📊 Evaluation frameworks

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 Citation

If you use Qyuzi in your research, please cite:

```bibtex
@software{Qyuzi,
  title={Qyuzi: A Scalable Small Language Model with Advanced Cognitive Architecture},
  author={HexZo Skibidi Toilet Ahh},
  year={2024},
  url={https://github.com/HexZoNetwork/Qyuzi}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Flash Attention**
- **Rotary Embeddings**
- **Mixture of Experts**
- **Tiktoken**
- **PyTorch Team**

---

## 📧 Contact

For questions, collaborations, or just to chat about AGI:
- **Telegram**: [@HexZo_Not_Devz](https://t.me/HexZo_Not_Devz)
---

<div align="center">

**Qyuzi never sleeps. 🏎️💨**

*Built with 🧠 and ⚡ by A Single Person [just a kid]*


</div>

