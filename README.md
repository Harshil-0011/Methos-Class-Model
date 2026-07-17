# Methos Class Model

Train a custom code-focused LLM from scratch on **4× A100 80GB** using the
**Methos Class Model** — an NSLT-derived architecture with
**O(1) memory** w.r.t. sequence length.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia">
  <img alt="Tests" src="https://img.shields.io/badge/tests-109%20passing-brightgreen">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Launch training (pretrain → SFT → instruction tuning)
bash scripts/train_4gpu.sh
```

No interaction needed. Training auto-resolves the tokenizer, streams datasets,
and saves checkpoints every 1,000 steps.

---

## What is the Methos Class Model?

A **non-Transformer** LLM derived from the NSLT architecture that replaces self-attention with four custom layers:

```
Tokens → [SSM Compression] → [LTC Routing] → [Latent Sandbox] → [Sparse Output] → Next Token
           O(1) memory        Neural ODE        K reasoning paths   1% vocab gating
```

| Transformer | Methos Class Model |
|---|---|
| O(T²) attention or O(T·d_kv) KV-cache | **O(1)** compressed state — fixed size, any length |
| Long context needs tricks (YaRN, ALiBi) | Native O(1) — state size independent of T |
| MatMul-heavy (attention + FFN) | Scan-heavy (SSM recurrence + ODE) |

---

## Features

- **7 CLI commands** — train, generate, test, benchmark, download tokenizer, validate config, system info
- **3-stage training** — pretrain (500K) → SFT (50K) → instruction tuning (20K)
- **FSDP full-shard** across 4 GPUs (ZeRO-3), 320 GB pooled VRAM
- **9 benchmarks** — HumanEval, MBPP, MMLU, GSM8K, HellaSwag, ARC, TruthfulQA, Winogrande, BBH
- **109 tests** — all passing
- **Claude-grade tokenizer** — `Xenova/claude-tokenizer` (BPE, ~100K vocab)
- **5 SSM scan backends** — sequential, vectorized, Triton, TorchScript JIT, CUDA

---

## CLI Overview

| Command | What it does |
|---|---|
| `python main.py full-training` | Run pretrain → SFT → instruction tuning |
| `python main.py generate --prompt "..."` | Generate text from a checkpoint |
| `python main.py test` | Run the 109-test suite |
| `python main.py benchmark` | Run benchmarks against a checkpoint |
| `python main.py download-tokenizer` | Download a HuggingFace tokenizer |
| `python main.py config-validate` | Validate `config.yaml` |
| `python main.py info` | Print system info |

### Training

```bash
# 4-GPU (production)
bash scripts/train_4gpu.sh

# Single GPU (debug)
python main.py full-training

# Fresh start (ignore checkpoints)
python main.py full-training --fresh-start
```

Training auto-resumes from the latest checkpoint. Each stage uses its own
learning rate, batch size, and optimizer:

| Stage | Steps | LR | Batch | Optimizer |
|---|---|---|---|---|
| Pretrain | 500K | 2e-4 | 64 | adamw_fused |
| SFT | 50K | 5e-6 | 16 | adamw_8bit |
| Instruction Tuning | 20K | 1e-5 | 16 | adamw_fused |

### Generation

```bash
python main.py generate --prompt "Write a Python HTTP server"
echo "def fibonacci(n):" | python main.py generate

python main.py generate \
  --prompt "Write a Rust HTTP server" \
  --checkpoint models/methos/sft \
  --max-new-tokens 2048 \
  --temperature 0.8 \
  --top-p 0.95
```

### Testing

```bash
python main.py test                          # all 109
python main.py test --filter ssm             # SSM scan tests only
python -m pytest tests/ -v --tb=short -x      # verbose, stop on first failure
python -m pytest tests/ --cov=src            # coverage
```

### Benchmarking

```bash
python main.py benchmark                                          # all
python main.py benchmark --benchmarks "human_eval,mbpp"           # specific
python main.py benchmark --checkpoint models/methos/best          # custom checkpoint
```

### Tokenizer

```bash
python main.py download-tokenizer                                 # Claude tokenizer
python main.py download-tokenizer --model-id google/gemma-2-27b-it
python main.py download-tokenizer --force
```

---

## Project Layout

```
.
├── main.py                     # CLI entrypoint (7 commands)
├── config.yaml                 # Training configuration
├── scripts/train_4gpu.sh       # 4-GPU torchrun launcher
├── src/
│   ├── nslt/                   # Methos Class Model architecture (pure PyTorch)
│   │   ├── model.py            # NSLTModel (~694 lines)
│   │   ├── layer1_ssm.py       # SSM compression engine
│   │   ├── layer2_ltc.py       # Neural ODE routing
│   │   ├── layer3_sandbox.py   # Latent sandbox reasoning
│   │   ├── layer4_output.py    # Sparse output gating
│   │   ├── ssm_scan.py         # 5 scan backends + autograd
│   │   ├── moe_ssm.py          # MoE SSM blocks
│   │   ├── mcts_sandbox.py     # MCTS latent sandbox
│   │   ├── multiscale_ssm.py   # Multi-scale hierarchy
│   │   └── vision_encoder.py   # SigLIP vision encoder
│   ├── config/schema.py        # Pydantic config (463 lines)
│   ├── models/factory.py       # Model creation + loading
│   ├── training/pipeline.py    # Training orchestration
│   ├── tokenizer_trainer.py    # Tokenizer download + BPE training
│   ├── infrastructure/
│   │   ├── distributed.py      # FSDP distributed setup
│   │   └── tracking.py         # WandB/MLflow/TensorBoard
│   └── data/pipeline.py        # Data streaming + processing
├── tests/                      # 109 tests
└── hf_cache/                   # Dataset cache
```

For the complete reference (every config field, every CLI flag, architecture
deep-dive, troubleshooting guide, benchmark methodology), see
[`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md).

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and CUDA 12.1+ for GPU training.

---

## Test Suite

109 tests across 11 files:

```
tests/test_nslt.py               # 17 — All 4 layers + full model
tests/test_integration.py         # 12 — SSM scan, training, MoE, vision, MCTS
tests/test_alignment.py           # 4 — DPO/ORPO/SimPO loss
tests/test_config.py              # 7 — Config validation + migration
tests/test_data_collector.py      # 3 — Dataset streaming
tests/test_data_pipeline.py       # 20 — Text cleaning, quality, dedup
tests/test_evaluation.py          # 4 — Safety, benchmarks
tests/test_generation.py          # 8 — Code extraction, language aliasing
tests/test_quality.py             # 6 — Quality scoring, contamination
tests/test_trainer.py             # 3 — Prompt formatting, tokenization
tests/test_validation.py          # 10 — Python syntax + execution
```

---

## Benchmarks

| Benchmark | Type | Problems | Metric |
|---|---|---|---|
| HumanEval | Code generation | 164 | pass@1 |
| MBPP | Programming tasks | 417 | pass@1 |
| MMLU | Knowledge (57 subjects) | ~14K | accuracy |
| HellaSwag | Commonsense NLI | 10K | accuracy |
| ARC | Science QA | 2,590 | accuracy |
| GSM8K | Math word problems | 1,319 | exact match |
| TruthfulQA | Factuality | 817 | mc1/mc2 |
| Winogrande | Coreference | 1,267 | accuracy |
| BBH | Reasoning (23 tasks) | 6,511 | accuracy |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| CUDA OOM | Reduce `batch_size`, lower `max_seq_length`, enable `expandable_segments` |
| NCCL errors | Set `NCCL_DEBUG=WARN`, `NCCL_NVLS_ENABLE=0`, `CUDA_DEVICE_MAX_CONNECTIONS=1` |
| "Tokenizer not found" | Run `python main.py download-tokenizer` |
| Architecture mismatch | `rm -rf models/methos/checkpoints; bash scripts/train_4gpu.sh --fresh-start` |
| "No module named 'src'" | Run from the project root directory |
| Training hangs at init | Check GPUs with `nvidia-smi`, kill stale `torchrun` processes |

---

## License

MIT
