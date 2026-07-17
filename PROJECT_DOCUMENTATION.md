# NSLT-Zero — Agent Reference

> **Purpose:** This document is the single source of truth for AI agents working on this codebase.
> Every config field, CLI flag, architecture detail, data flow, error state, and edge case is
> documented here. If an agent needs to modify the code, read this first.

---

## Table of Contents

- [1. System Overview](#1-system-overview)
- [2. Entry Points](#2-entry-points)
- [3. Configuration System](#3-configuration-system)
  - [3.1 Schema Definition](#31-schema-definition)
  - [3.2 Config File Location](#32-config-file-location)
  - [3.3 All Fields Reference](#33-all-fields-reference)
- [4. CLI Commands](#4-cli-commands)
  - [4.1 full-training](#41-full-training)
  - [4.2 generate](#42-generate)
  - [4.3 test](#43-test)
  - [4.4 benchmark](#44-benchmark)
  - [4.5 download-tokenizer](#45-download-tokenizer)
  - [4.6 config-validate](#46-config-validate)
  - [4.7 info](#47-info)
- [5. Training Pipeline](#5-training-pipeline)
  - [5.1 Startup Sequence](#51-startup-sequence)
  - [5.2 Stage Execution](#52-stage-execution)
  - [5.3 FSDP Details](#53-fsdp-details)
  - [5.4 Checkpointing](#54-checkpointing)
  - [5.5 Error Handling](#55-error-handling)
- [6. Model Architecture](#6-model-architecture)
  - [6.1 NSLTModel](#61-nsltmodel)
  - [6.2 Layer 1: SSMCompressionEngine](#62-layer-1-ssmcompressionengine)
  - [6.3 Layer 2: LTCRoutingLayer](#63-layer-2-ltcroutinglayer)
  - [6.4 Layer 3: LatentSandbox](#64-layer-3-latentsandbox)
  - [6.5 Layer 4: SparseOutputSynthesizer](#65-layer-4-sparseoutputsynthesizer)
  - [6.6 SSM Scan Backends](#66-ssm-scan-backends)
  - [6.7 MoE SSM Block](#67-moe-ssm-block)
- [7. Tokenizer](#7-tokenizer)
  - [7.1 Sources](#71-sources)
  - [7.2 Loader Fallback Chain](#72-loader-fallback-chain)
  - [7.3 Special Tokens](#73-special-tokens)
- [8. Data Pipeline](#8-data-pipeline)
- [9. Distributed Setup](#9-distributed-setup)
- [10. Source Files](#10-source-files)
  - [10.1 src/ directory tree](#101-src-directory-tree)
  - [10.2 Key file summaries](#102-key-file-summaries)
- [11. Test Suite](#11-test-suite)
- [12. Benchmarks](#12-benchmarks)
- [13. Known Flaky Tests & Edge Cases](#13-known-flaky-tests--edge-cases)

---

## 1. System Overview

- **Language:** Python 3.10+
- **Framework:** PyTorch 2.6, HuggingFace Transformers 4.48+, Datasets 3.3+
- **Distributed:** FSDP full-shard (ZeRO-3), NCCL backend
- **Hardware target:** 4× A100 80GB (320 GB pooled VRAM)
- **Config:** Pydantic v2 `BaseModel` in `src/config/schema.py`
- **CLI:** `argparse` in `main.py`, 7 subcommands
- **Tests:** pytest, 109 tests, all pass

---

## 2. Entry Points

| File | Purpose |
|---|---|
| `main.py` | CLI dispatcher. Defines `ArgumentParser` with 7 subcommands. Calls into services. |
| `src/training/pipeline.py` | `TrainingPipeline` — orchestrates 3-stage training. |
| `src/models/factory.py` | `ModelFactory` — create, load, save models. |
| `src/tokenizer_trainer.py` | Tokenizer download (`download_tokenizer`) and BPE training (`train_custom_tokenizer`). |
| `src/infrastructure/distributed.py` | `DistributedSetup` — process group init, device management, FSDP config. |
| `src/config/schema.py` | All pydantic models: `Config`, `ModelArchitectureConfig`, `TrainingConfig`, etc. |
| `src/nslt/model.py` | `NSLTModel` — the core neural architecture. |

---

## 3. Configuration System

### 3.1 Schema Definition

Defined in `src/config/schema.py`. The root model is `Config`. All validation
is automatic via pydantic. The schema includes a `@model_validator(mode="before")`
that migrates v1 config keys to v2:

```python
# v1 → v2 auto-migration:
# data_collection.*        → data.*
# fsdp.*                   → distributed.fsdp.*
# rope_scaling_factor      → rope_scaling.factor
# rope_scaling_type        → rope_scaling.type
# languages key            → removed
```

### 3.2 Config File Location

- Default: `config.yaml` in project root.
- Read by: `main.py` → `load_config("config.yaml")`.
- The path is hardcoded; agents should not change it without updating all callers.

### 3.3 All Fields Reference

```yaml
# ── Top-Level ──
project:
  name: NSLT-Zero                    # str — Project name for logging
  seed: 42                           # int — Random seed (set_seed() called in pipeline init)

model:
  name: NSLT-Zero                    # str — Display name
  dtype: bfloat16                    # "bfloat16" | "float16" | "float32"
  device: auto                       # "auto" → resolved to "cuda" or "cpu" at runtime
  train_from_scratch: true           # bool — If false, would try pretrained (not implemented for NSLT)
  architecture:                      # ModelArchitectureConfig (nested)
  load_in_8bit: false                # bool — Not implemented for NSLT
  load_in_4bit: false                # bool — Not implemented for NSLT

# ── Architecture ──
model.architecture:
  model_type: nslt                   # "nslt" | "llama" | "mixtral" | "qwen2_moe" | "deepseek_v2"
  hidden_size: 7168                  # int — d_model
  num_hidden_layers: 56              # int — Only used for non-NSLT model types
  num_attention_heads: 56            # int — Only used for non-NSLT model types
  num_key_value_heads: 8             # int — GQA heads (non-NSLT)
  intermediate_size: 18432           # int — FFN intermediate (non-NSLT)
  max_position_embeddings: 16384     # int — Absolute max sequence length
  rope_theta: 10000000.0             # float — RoPE base frequency
  rope_scaling:
    type: yarn                       # "yarn" | "linear" | "dynamic"
    factor: 16.0                     # float
    target_max_length: 262144        # int
    original_max_position_embeddings: 16384  # int
  attention_implementation: flash_attention_2  # "sdpa" | "flash_attention_2" | "eager"
  tie_word_embeddings: false         # bool
  attention_bias: false              # bool
  attention_dropout: 0.0             # float
  hidden_act: silu                   # "silu" | "gelu" | "relu"
  rms_norm_eps: 1e-6                # float
  initializer_range: 0.02            # float
  pretraining_tp: 1                  # int — Reserved
  mlp_bias: false                    # bool
  gradient_checkpointing: true       # bool
  use_compile: false                 # bool — torch.compile, experimental
  vocab_size: 128000                 # int — Must match tokenizer.vocab_size or len(tokenizer)

  # MoE sub-config (ignored for nslt model_type)
  moe:
    num_experts: 8
    top_k: 2
    expert_capacity: null
    shared_expert_count: 1
    shared_expert_gate: true
    norm_topk_prob: true
    output_router_logits: false
    aux_loss_coef: 0.01
    jitter_noise: 0.0
    router_aux_loss_coef: 0.001

  # NSLT sub-config (ignored for non-nslt model_type)
  nslt:
    d_state: 2048                    # int — SSM compressed state dimension
    d_hidden: 7168                   # int — LTC + sandbox hidden dimension
    n_ssm_layers: 4                  # int — Number of stacked SSM blocks
    n_ode_steps: 8                   # int — RK4 integration steps per token
    solver: rk4                      # "euler" | "rk4" | "adjoint"
    n_trajectories: 8                # int — Parallel sandbox trajectories
    n_sim_steps: 16                  # int — Energy descent steps
    use_efficient_sandbox: false     # bool — Memory-efficient variant
    sparsity_pct: 1.0               # float — % of vocabulary activated per token

  # Vision sub-config (not fully implemented for training)
  multimodal:
    vision:
      enabled: false
      vision_encoder: google/siglip-so400m-patch14-384
      image_size: 384
      patch_size: 14
      vision_hidden_size: 1152
      num_vision_layers: 27
      num_attention_heads: 16
      intermediate_size: 4304
      projection_dim: 5120
      freeze_vision_encoder: true
      tie_vision_embeddings: false
      image_token_id: 128000
      max_images_per_sample: 5

# ── Training ──
training:
  max_seq_length: 8192               # int — Sequences longer than this are truncated
  response_only_loss: true           # bool — Mask prompt tokens with -100 in labels
  ignore_data_skip: true             # bool — Resume without re-skipping data
  save_steps: 1000                   # int — Checkpoint interval
  save_total_limit: 5                # int — Keep last N checkpoints
  eval_strategy: steps               # "steps" | "epoch" | "no"
  eval_steps: 500                    # int — Evaluation interval
  logging_steps: 10                  # int — Logging interval

  pretrain:
    enabled: true
    learning_rate: 2e-4
    lr_scheduler_type: cosine
    warmup_steps: 5000
    weight_decay: 0.1
    batch_size: 2                    # Per GPU
    gradient_accumulation_steps: 8
    max_steps: 500000
    optimizer: adamw_fused           # "adamw" | "adamw_8bit" | "adamw_fused" | "sgd"
    data_mix: {code: 0.3, web_text: 0.3, books: 0.1, math: 0.1, science: 0.1, other: 0.1}

  sft:
    enabled: true
    learning_rate: 5e-6
    lr_scheduler_type: cosine
    warmup_steps: 500
    weight_decay: 0.05
    batch_size: 1
    gradient_accumulation_steps: 4
    max_steps: 50000
    optimizer: adamw_8bit
    data_mix: {sft_instructions: 0.5, conversations: 0.3, code_tasks: 0.2}

  instruction_tuning:
    enabled: true
    learning_rate: 1e-5
    lr_scheduler_type: cosine
    warmup_steps: 200
    weight_decay: 0.05
    batch_size: 1
    gradient_accumulation_steps: 4
    max_steps: 20000
    optimizer: adamw_fused

  alignment:
    enabled: false
    methods: [dpo, kto]
    method_configs:
      dpo:
        learning_rate: 3e-7
        beta: 0.1
        batch_size: 4
        max_steps: 2000
        warmup_steps: 100
        label_smoothing: 0.0
        loss_type: sigmoid
      orpo:
        learning_rate: 3e-7
        beta: 0.05
        batch_size: 4
        max_steps: 2000
        warmup_steps: 100
      simpo:
        learning_rate: 3e-7
        gamma: 0.5
        beta: 2.0
        batch_size: 4
        max_steps: 2000
        warmup_steps: 100
      kto:
        learning_rate: 3e-7
        beta: 0.1
        batch_size: 4
        max_steps: 2000
        warmup_steps: 100
        desirable_weight: 1.0
        undesirable_weight: 1.0

  rlhf:
    enabled: false
    learning_rate: 1e-6
    batch_size: 4
    max_steps: 10000
    ppo_epochs: 4
    kl_coef: 0.05
    cliprange: 0.2
    vf_coef: 0.1

  safety:
    enabled: false
    safety_training_steps: 5000
    harmlessness_loss_coef: 0.1
    refusal_training: true
    honesty_training: true
    constitution_training: true
    red_teaming_iters: 1000
    red_team_model: null

# ── Distributed ──
distributed:
  strategy: fsdp                   # "fsdp" | "deepspeed" | "ddp" | "none"
  fsdp:
    enabled: true
    sharding_strategy: full_shard  # "full_shard" | "hybrid_shard" | "no_shard"
    transformer_layer_cls: NSLTModel  # FSDP wrapping target
    backward_prefetch: backward_pre   # "backward_pre" | "backward_post" | "no_prefetch"
    forward_prefetch: true
    activation_checkpointing: true
    use_orig_params: true
    sync_module_states: true
    limit_all_gathers: true
    mixed_precision: bf16          # "bf16" | "fp16" | "fp32"
  deepspeed:
    zero_stage: 3
    offload_optimizer: cpu          # null | "cpu" | "nvme"
    offload_params: null            # null | "cpu" | "nvme"

# ── Data ──
data:
  cache_dir: ./hf_cache
  streaming: true
  max_cache_gb: 200
  num_download_workers: 16
  datasets:
    - path: codeparrot/codeparrot-clean
      max_samples: 100000
      split: train
      name: null
      data_dir: null
      language: null
      category: null
      weight: 1.0
  quality:
    min_length: 30
    max_length: 500000
    deduplication:
      enabled: true
      method: minhash               # "exact" | "minhash" | "embedding"
      threshold: 0.85
    contamination:
      enabled: true
      benchmarks: [human_eval, mbpp, mmlu, gsm8k, arc, hellaswag, truthfulqa]
    quality_scoring:
      enabled: true
      method: heuristic             # "heuristic" | "perplexity" | "classifier"
    language_detection:
      enabled: true
    toxicity_filtering:
      enabled: true
  curriculum:
    enabled: false
    stages: []

# ── Tokenizer ──
tokenizer:
  source: huggingface               # "huggingface" | "custom"
  huggingface_model: Xenova/claude-tokenizer
  vocab_size: 128000                # Used only when source: custom
  max_samples: 100000               # Used only when source: custom
  type: bpe                         # "bpe" | "unigram" | "wordpiece"
  force: false                      # Force re-download / re-train
  add_prefix_space: false
  add_bos_token: true
  add_eos_token: true

# ── Output ──
output:
  model_dir: ./models/fable5
  data_dir: ./data
  checkpoint_dir: ./models/fable5/checkpoints
  log_dir: ./logs
  experiment_tracking:
    enabled: false
    provider: none                  # "wandb" | "mlflow" | "tensorboard" | "none"
    project: fable-5

# ── Generation defaults ──
generation:
  max_new_tokens: 32768
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  repetition_penalty: 1.05
  do_sample: true
  num_beams: 1

# ── Evaluation ──
evaluation:
  benchmarks: [mmlu, hellaswag, arc, human_eval, mbpp, gsm8k, truthfulqa, winogrande, bbh]
  benchmark_configs: {}
  automated_report: true
  report_dir: ./eval_reports
  timeout: 60
  evaluate_during_training: true
  eval_frequency: 5000
```

---

## 4. CLI Commands

Defined in `main.py`. Uses `argparse` with subparsers. All commands support `-h`.

### 4.1 full-training

```python
# main.py dispatch:
# if args.command == "full-training" → cmd_full_training(args)
#   → TrainingPipeline(cfg).initialize(fresh_start=args.fresh_start, resume_checkpoint=...)
#   → pipeline.full_training_sequence()
#   → pipeline.cleanup()  # always runs (try/finally)
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--fresh-start` | `store_true` | `False` | Skip checkpoint resume; start from scratch |

**Startup sequence** (in order):
1. `load_config("config.yaml")` — validates and migrates
2. `DistributedSetup(cfg)` — reads env vars `WORLD_SIZE`, `LOCAL_RANK`, `RANK`;
   calls `torch.cuda.set_device(local_rank)` then `dist.init_process_group(backend="nccl")`
3. `ensure_tokenizer("models/tokenizer", "config.yaml")` — rank 0 only, then `dist.barrier()`
4. `ModelFactory.load_tokenizer("models/tokenizer")` — loads saved tokenizer
5. `ModelFactory.create_model(cfg, tokenizer)` or `load_model()` for resume
6. `DataPipeline(cfg, tokenizer)` — initializes data streaming
7. Run each enabled stage

**Stage execution** (`_load_stage_data`):
- Calls `self.data_pipeline.collector.stream_samples(limit=None)`
- Builds dataset from samples via `build_stage_dataset`
- Creates `Trainer` via `_build_trainer` with per-stage config
- Calls `trainer.train()`, saves checkpoint, saves tokenizer

**Cleanup** (always):
- `self.tracker.finish()`
- `gc.collect()`
- `torch.cuda.empty_cache()`
- `dist.destroy_process_group()` — prevents "already initialized" on re-run

### 4.2 generate

```python
# main.py dispatch:
# if args.command == "generate" → cmd_generate(args)
#   → ModelFactory.load_model(checkpoint_path, cfg, tokenizer)
#   → model.generate(**gen_kwargs)
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--prompt` | `str` | `None` | Input prompt. Reads from stdin if omitted. |
| `--checkpoint` | `str` | `models/fable5` | Path to model directory |
| `--tokenizer` | `str` | `models/tokenizer` | Path to tokenizer directory |
| `--max-new-tokens` | `int` | `1024` | Max tokens to generate |
| `--temperature` | `float` | `0.7` | Sampling temperature; 0 = greedy |
| `--top-p` | `float` | `0.9` | Nucleus sampling threshold |
| `--top-k` | `int` | `40` | Top-k sampling |

**Model loading logic** (`ModelFactory.load_model`):
1. Check `path/config.json` for `model_type`
2. If `"nslt"` → construct `NSLTModel`, load `pytorch_model.bin` via `state_dict`
3. If other → `AutoModelForCausalLM.from_pretrained(path, ...)`
4. If no tokenizer provided → `ModelFactory.load_tokenizer()`

**Tokenizer loading** (`_try_load_tokenizer`, 3-tier fallback):
1. `AutoTokenizer.from_pretrained(path, trust_remote_code=False)`
2. `PreTrainedTokenizerFast.from_pretrained(path)`
3. Manual BPE from `vocab.json` + `merges.txt`

**pad_token fix:** If `tokenizer.pad_token is None`, calls
`tokenizer.add_special_tokens({"pad_token": "<pad>"})`.

### 4.3 test

| Flag | Type | Default | Description |
|---|---|---|---|
| `--filter` | `str` | `None` | `-k` filter passed to pytest |

```python
# Implementation:
# cmd = ["pytest", "tests/", "-v"]
# if args.filter: cmd.extend(["-k", args.filter])
# subprocess.run(cmd)
```

### 4.4 benchmark

| Flag | Type | Default | Description |
|---|---|---|---|
| `--checkpoint` | `str` | `models/fable5` | Model path |
| `--tokenizer` | `str` | `models/tokenizer` | Tokenizer path |
| `--benchmarks` | `str` | `None` | Comma-separated names |

If `--benchmarks` is omitted, runs the default benchmark list `human_eval,mbpp`. Runs via
`BenchmarkRunner(model, tokenizer).run_benchmarks(benchmark_list)`.

### 4.5 download-tokenizer

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model-id` | `str` | `Xenova/claude-tokenizer` | HF model ID |
| `--output` | `str` | `models/tokenizer` | Output directory |
| `--force` | `store_true` | `False` | Overwrite existing |

Uses `AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)`.
Always ensures `bos_token`, `eos_token`, `unk_token`, `pad_token`, `mask_token`
are set. Saves with `tokenizer.save_pretrained(output_path)`.

### 4.6 config-validate

No flags. Reads `config.yaml` via `load_config()`. Prints a validation summary on success, error message on failure. Exits with code 0 on success, 1 on failure.

**Exception handling:** Only catches `(yaml.YAMLError, ValidationError, FileNotFoundError)`.
`KeyboardInterrupt` and `SystemExit` propagate.

### 4.7 info

No flags. Prints:
- Python version (`sys.version`)
- PyTorch version (`torch.__version__`)
- CUDA available + version
- GPU count, names, free memory per device
- World size from env
- Distributed backend (nccl/gloo)

---

## 5. Training Pipeline

### 5.1 Startup Sequence

```
main.py: cmd_full_training()
  ├─ load_config("config.yaml")           → Config object
  ├─ TrainingPipeline(cfg)
  │   ├─ DistributedSetup(cfg)
  │   │   ├─ torch.cuda.set_device(local_rank)
  │   │   └─ dist.init_process_group(backend="nccl")
  │   ├─ ExperimentTracker(cfg)
  │   └─ set_seed(42)
  ├─ pipeline.initialize(fresh_start, resume_checkpoint)
  │   ├─ ensure_tokenizer("models/tokenizer", "config.yaml")
  │   │   ├─ if source=="huggingface": download_tokenizer()
  │   │   └─ if source=="custom": train_custom_tokenizer()
  │   ├─ ModelFactory.load_tokenizer("models/tokenizer")
  │   ├─ if fresh_start or no checkpoint:
  │   │   └─ ModelFactory.create_model(cfg, tokenizer) → model
  │   ├─ else:
  │   │   ├─ ModelFactory.is_compatible(cfg, tokenizer, checkpoint)
  │   │   └─ ModelFactory.load_model(checkpoint, cfg, tokenizer)
  │   ├─ DataPipeline(cfg, tokenizer)
  │   └─ AlignmentPipeline(model, tokenizer, cfg)
  └─ pipeline.full_training_sequence()
      ├─ _load_stage_data("pretrain")     → Dataset
      ├─ run_pretrain(dataset)
      ├─ _load_stage_data("sft")          → Dataset
      ├─ run_sft(dataset)
      ├─ _load_stage_data("instruction_tuning") → Dataset
      └─ run_instruction_tuning(dataset)
```

### 5.2 Stage Execution

Each `_train_stage(dataset, stage_name, stage_cfg)`:
1. Resolves optimizer via `optim_map`:
   - `"adamw"` → `"adamw_torch"`
   - `"adamw_8bit"` → `"paged_adamw_8bit"` (or `"adamw_torch"` fallback)
   - `"adamw_fused"` → `"adamw_torch_fused"` (or `"adamw_torch"` fallback)
   - `"sgd"` → `"sgd"`
2. Creates `TrainingArguments` with per-stage hyperparams
3. Creates `Trainer(model, args, train_dataset, data_collator, tokenizer, callbacks)`
4. Calls `trainer.train()`
5. Saves model to `output_dir / stage_name`
6. Saves tokenizer to same directory
7. Returns metrics dict

**Key `TrainingArguments` settings:**
- `save_only_model=True` — only saves model weights, not optimizer state
- `ddp_find_unused_parameters=False` — avoids expensive param scan
- `remove_unused_columns=False` — keeps all dataset columns
- `report_to` → from `experiment_tracking.provider`

### 5.3 FSDP Details

**In `distributed.py` `get_training_args()`:**
```python
args["fsdp"] = "full_shard auto_wrap"
args["fsdp_config"] = {
    "transformer_layer_cls_to_wrap": [fsdp_cfg.transformer_layer_cls],
    "backward_prefetch": fsdp_cfg.backward_prefetch,
    "forward_prefetch": fsdp_cfg.forward_prefetch,
    "activation_checkpointing": fsdp_cfg.activation_checkpointing,
    "use_orig_params": fsdp_cfg.use_orig_params,
    "sync_module_states": fsdp_cfg.sync_module_states,
    "limit_all_gathers": fsdp_cfg.limit_all_gathers,
    "fsdp_mixed_precision": fsdp_cfg.mixed_precision,  # string, not bool
}
```

**Override in `pipeline.py` `_build_trainer()`:**
```python
fsdp_config["transformer_layer_cls_to_wrap"] = [ARCH_FSDP_LAYER_MAP.get(model_type, "LlamaDecoderLayer")]
```
This ensures the correct layer class is used based on `model_type`.

**Key constraint:** For NSLT, `transformer_layer_cls_to_wrap = ["NSLTModel"]`
wraps the entire model as one FSDP unit. This is intentional — NSLT has
non-repeated layers of different types, so per-layer wrapping isn't meaningful.

### 5.4 Checkpointing

- **Interval:** `training.save_steps` (default 1000)
- **Retention:** `training.save_total_limit` (default 5)
- **Per-stage directories:** `models/fable5/<stage_name>/`
- **Auto-resume:** Pipeline checks `models/fable5/checkpoints` for latest.
  Compares architecture spec between saved `config.json` and current config.
  On mismatch → log warning, create fresh model.
- **Save format:** `model.save_pretrained(dir, safe_serialization=True)`
  + `tokenizer.save_pretrained(dir)`

### 5.5 Error Handling

- `try/finally` ensures `cleanup()` always runs (destroy process group, empty cache)
- `_load_stage_data` catches `Exception` and returns `None` (skips stage)
- `_build_trainer` uses `getattr(stage_cfg, "warmup_steps", 200)` fallbacks for
  optional fields
- `destroy_process_group()` guards with `if dist.is_initialized()`
- If training crashes mid-stage, the next launch resumes from the latest
  checkpoint (optimizer state may be lost, but model weights are preserved)

---

## 6. Model Architecture

### 6.1 NSLTModel

**File:** `src/nslt/model.py` (~694 lines)

```python
class NSLTModel(nn.Module):
    def __init__(self, vocab_size, d_model, d_state, d_hidden, n_ssm_layers,
                 max_seq_len, rope_base, sparsity_pct, n_ode_steps=8,
                 n_trajectories=8, n_sim_steps=16, use_efficient_sandbox=False,
                 dtype=torch.bfloat16):
```

**Components:**
- `TokenEmbedding(vocab_size, d_model)` — learned embeddings, scaled by `sqrt(d_model)`
- `RotaryPositionEncoding(d_model, max_seq_len, rope_base)` — RoPE
- `SSMCompressionEngine` × `n_ssm_layers` — stacked SSM blocks
- `LTCRoutingLayer(d_model, d_hidden, d_state, ...)` — neural ODE routing
- `LatentSandbox(d_hidden, d_state, n_trajectories, n_sim_steps)` — parallel reasoning
- `SparseOutputSynthesizer(d_hidden, vocab_size, sparsity_pct)` — ultra-sparse output

**Forward pass:**
1. `TokenEmbedding(input_ids)` → `[B, T, d_model]`
2. `RotaryPositionEncoding(x)` — applies RoPE in place
3. For each SSM layer: `x, h = ssm(x)` — compresses to `h [B, d_state]`
4. `LTCRoutingLayer(x, h)` → `[B, T, d_hidden]`
5. `LatentSandbox(z)` → `[B, d_hidden]` — select best trajectory
6. `SparseOutputSynthesizer(z)` → `[B, T, V]` logits (sparse)

**MoE variant:** `MoENSLTModel` in the same file wraps `NSLTModel` and adds
MoE routing on the SSM layers.

**meta device guard:** `self.to(device, dtype)` is guarded with
`if device.type != "meta"` to prevent RuntimeError when model is on meta device.

### 6.2 Layer 1: SSMCompressionEngine

**File:** `src/nslt/layer1_ssm.py` (~151 lines)

**Math:** `h_t = exp(∆_t·A)·h_{t-1} + (exp(∆_t·A)-I)/A·B_t·x_t`
**Output:** `y_t = C_t·h_t`

**Components:**
- `in_proj`: `Linear(d_model, d_inner*2)` — split into `x_inner` and `x_gate`
- `conv1d`: depthwise `Conv1d(d_inner, d_inner, kernel=4, padding=3)` — local mixing
- `SiLU` activation
- `dt_proj`: `Linear(d_inner, dt_rank)` → `F.softplus(...)` for positivity
- `A_log`: `Parameter [d_state]` — `A = -exp(A_log)` (negative diagonal, stable)
- `B_proj`, `C_proj`: `Linear(d_inner, dt_rank)` — input-dependent
- `out_proj`: `Linear(d_inner, d_model)` — project back
- `LayerNorm(d_model)` — pre-normalization

**Forward:**
1. `norm(x)` → `in_proj(x)` → chunk → `x_inner`, `x_gate`
2. `conv1d(x_inner.permute(...))` → slice to seq_len → permute back → SiLU
3. Compute `delta` from `dt_proj(x_conv)` + softplus + `dt_bias`
4. Compute `A = -exp(A_log)`, `B = B_proj(x_conv)`, `C = C_proj(x_conv)`
5. `selective_scan(x_conv, delta, A, B, C, dt_rank)` → `y, h_final`
6. `y * SiLU(x_gate)` → `out_proj(y)` → output

### 6.3 Layer 2: LTCRoutingLayer

**File:** `src/nslt/layer2_ltc.py` (~200 lines)

**Math:** `dz/dt = -[w_tau·σ(w_tau·z + b_tau)]·z + f(z, I(t))`

- **Liquid time-constant:** `τ(z) = 1 / (w_tau·σ(w_tau·z + b_tau))`
- **Solvers:** Euler (1st order), RK4 (4th order, default), adjoint (memory-efficient)
- **Input dynamics:** MLP that takes concatenated `[z, I(t)]` and produces `dz/dt`
- Runs `n_ode_steps` integration steps per input token

**Solver choice:**
| Solver | Order | Gradient Memory | Speed |
|---|---|---|---|
| `euler` | O(h) | Full (stores all intermediates) | Fastest |
| `rk4` | O(h⁴) | Full (stores 4 intermediates per step) | Fast |
| `adjoint` | O(h⁴) | O(1) (recomputes forward) | Slower |

### 6.4 Layer 3: LatentSandbox

**File:** `src/nslt/layer3_sandbox.py` (~300 lines)

- Creates K copies of state `z` → `z_1 ... z_K`
- Each evolves via gradient descent on `E(z)` for `n_sim_steps` iterations:
  ```python
  z_i = z_i - lr * torch.autograd.grad(E(z_i).sum(), z_i)[0]
  ```
- Selects trajectory with lowest final energy
- Energy = `||z - decoder(encoder(z))||² + λ₂·R(z)`

**Energy function network:**
- Encoder: `Linear(d_hidden, d_latent*2)` → SiLU → `Linear(d_latent*2, d_latent)`
- Decoder: `Linear(d_latent, d_latent*2)` → SiLU → `Linear(d_latent*2, d_hidden)`
- R-net: `Linear(d_hidden, d_hidden//4)` → SiLU → `Linear(d_hidden//4, 1)`

**Efficient variant:** `LatentSandboxEfficient` — batches all K trajectories
in a single forward pass (lower memory overhead).

**MCTS variant** (`src/nslt/mcts_sandbox.py`): Uses Monte Carlo Tree Search
instead of gradient descent. Each node is a latent state. UCB-based selection,
expansion, simulation, back-propagation. More exploration at higher compute cost.

### 6.5 Layer 4: SparseOutputSynthesizer

**File:** `src/nslt/layer4_output.py` (~236 lines)

**Adaptive top-k gating:**
```python
k = min_k + (max_k - min_k) * entropy / log(vocab_size)
```
Where `entropy = -Σ p_i·log(p_i)` of the gate distribution.

**Components:**
- `hidden_proj`: `Linear(d_hidden, d_model)` — project hidden to model dim
- `output_embedding`: `Parameter [vocab_size, d_model]` — tied embedding, not tied by default
- `logit_temperature`: `Parameter [1]` — learned temperature, abs() for positivity
- `gate`: MLP that produces logits over vocabulary
- `SparseGatingUnit`: adaptive top-k selector

**Forward (training):**
1. `h = hidden_proj(x)` — `[B, d_model]`
2. `gate_values, top_indices, _ = gate(h)` — select top-k vocab entries
3. `target_logit = target_emb · h / temp` — logit for target token
4. `selected_logits = selected_embs · h / temp` — logits for top-k
5. `all_logits = concat([selected_logits, target_logit.unsqueeze(1)])`
6. `log_probs = log_softmax(all_logits)` — normalization over [k+1]
7. Return `log_probs[:, -1]` — log prob of target token

**Forward (inference):**
1. Same gate selection (no target token available)
2. Score only the top-k selected vocabulary entries
3. Return logits over [k] — never materialize [V]

### 6.6 SSM Scan Backends

**File:** `src/nslt/ssm_scan.py` (~445 lines)

```python
def selective_scan(x, delta, A, B, C, dt_rank, mode=None, use_autograd=False):
```

| Mode | Function | Hardware | Description |
|---|---|---|---|
| `sequential` | `selective_scan_sequential` | CPU/CUDA | Python for-loop, correct gradients |
| `vectorized` | `selective_scan_vectorized` | CUDA | Parallel prefix scan via cumsum |
| `triton` | `selective_scan_triton` | CUDA+Triton | Custom Triton kernel (fastest) |
| `jit` | `selective_scan_jit` | CPU | TorchScript JIT-compiled loop |
| `auto` | dispatches | Any | Triton→vectorized→sequential fallback |

**Dispatch logic (`auto` mode):**
```python
if x.is_cuda and _HAS_TRITON:
    return selective_scan_triton(...)
if x.is_cuda:
    return selective_scan_vectorized(...)
return selective_scan_sequential(...)
```

**Custom autograd Function** (`SSMScanFunction`):
- Forward saves `h_seq` for non-Triton modes (recomputed for Triton)
- Backward recomputes `h_t` per timestep for correct B/C/delta gradients
- Gradient shapes: `grad_x [B,T,d_inner]`, `grad_delta [B,T,dt_rank]`,
  `grad_A [d_state]`, `grad_B [B,T,dt_rank]`, `grad_C [B,T,dt_rank]`

### 6.7 MoE SSM Block

**File:** `src/nslt/moe_ssm.py` (~200 lines)

Wraps `SSMCompressionEngine` with mixture-of-experts routing:
- Router network selects top-k experts per token
- Each expert is an independent `SSMCompressionEngine`
- Output is weighted sum of selected experts' outputs
- Auxiliary load-balancing loss encourages uniform expert usage

---

## 7. Tokenizer

### 7.1 Sources

| `source` | Behavior | File created |
|---|---|---|
| `huggingface` | `AutoTokenizer.from_pretrained(model_id)` | `tokenizer.json`, `tokenizer_config.json` |
| `custom` | Train `ByteLevelBPETokenizer` on dataset corpus | Same |

### 7.2 Loader Fallback Chain

In `ModelFactory._try_load_tokenizer(path)`:

```python
try:
    return AutoTokenizer.from_pretrained(path, trust_remote_code=False)
except:
    try:
        return PreTrainedTokenizerFast.from_pretrained(path)
    except:
        # Manual BPE from raw vocab.json + merges.txt
        if (path/"vocab.json").exists() and (path/"merges.txt").exists():
            backend = ByteLevelBPETokenizer(str(vocab), str(merges))
            return PreTrainedTokenizerFast(
                tokenizer_object=backend._tokenizer,
                bos_token="<s>", eos_token="</s>",
                unk_token="<unk>", pad_token="<pad>",
            )
        raise RuntimeError(f"Unable to load tokenizer from {path}")
```

### 7.3 Special Tokens

Always added if missing after loading:
- `bos_token`: `<s>`
- `eos_token`: `</s>`
- `unk_token`: `<unk>`
- `pad_token`: `<pad>`
- `mask_token`: `<mask>`

**pad_token fix (critical):** If `tokenizer.pad_token is None` after loading,
`tokenizer.add_special_tokens({"pad_token": "<pad>"})` is called. This ensures
the data collator and loss computation have a valid padding token ID.

---

## 8. Data Pipeline

**File:** `src/data/pipeline.py` (~241 lines)

`DataPipeline` wraps `MassiveDataCollector` from `src/data/streaming.py`.

**Key flows:**
1. `DataPipeline(cfg, tokenizer)` → creates `MassiveDataCollector(cfg.data.datasets)`
2. `collector.stream_samples(limit=None)` → yields dicts with keys: `language`,
   `instruction`, `input`, `output`, etc.
3. `build_stage_dataset(samples)` → tokenizes and returns HuggingFace `Dataset`

**Quality filtering** (applied per-sample):
1. Length check: `30 ≤ len(content) ≤ 500000`
2. Low-quality marker check (`is_high_quality_content`): rejects samples
   containing "lorem ipsum", "todo: add code", "your code here", etc.
3. Contamination check: rejects samples matching benchmark patterns
4. Deduplication (minhash with configurable threshold)

---

## 9. Distributed Setup

**File:** `src/infrastructure/distributed.py` (~108 lines)

`DistributedSetup` class:

```python
class DistributedSetup:
    def __init__(self, cfg):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.is_distributed = self.world_size > 1
        if self.is_distributed:
            torch.cuda.set_device(self.local_rank)
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
```

**Critical details:**
- `torch.cuda.set_device(local_rank)` is called BEFORE `init_process_group`
  to ensure NCCL buffers are allocated on the correct GPU
- `is_main_process()` checks `self.rank == 0` (global rank), NOT `local_rank == 0`
- `auto_device()` returns `cuda:{local_rank}` or `cpu`
- `get_training_args()` returns a dict, not a `TrainingArguments` object —
  the actual `TrainingArguments` is constructed in `pipeline.py` `_build_trainer()`

---

## 10. Source Files

### 10.1 src/ directory tree

```
src/
├── __init__.py
├── model.py                   # SpecializedCoderModel (higher-level wrapper)
├── trainer.py                 # ModelTrainer (legacy trainer, kept for compatibility)
├── generator.py               # Code generation utilities
├── validator.py               # Python code validation
├── tokenizer_trainer.py       # Tokenizer download + BPE training
├── dataset.py                 # Dataset utilities
├── benchmark.py               # Legacy benchmark entry
├── massive_data_collector.py  # Legacy data collector
├── knowledge_graph.py         # Knowledge graph utilities
│
├── config/
│   ├── __init__.py
│   └── schema.py              # ~463 lines — ALL pydantic models
│
├── models/
│   ├── __init__.py
│   └── factory.py             # ~447 lines — ModelFactory
│
├── training/
│   ├── __init__.py
│   └── pipeline.py            # ~365 lines — TrainingPipeline
│
├── infrastructure/
│   ├── __init__.py
│   ├── distributed.py         # ~108 lines — DistributedSetup
│   └── tracking.py            # ~79 lines — ExperimentTracker
│
├── data/
│   ├── __init__.py
│   ├── pipeline.py            # ~241 lines — DataPipeline
│   ├── quality.py             # ~147 lines — QualityFilter, dedup, contamination
│   └── streaming.py           # MassiveDataCollector
│
├── alignment/
│   ├── __init__.py
│   └── pipeline.py            # DPO, ORPO, SimPO, KTO, safety training
│
├── evaluation/
│   ├── __init__.py
│   ├── benchmarks.py          # ~426 lines — BenchmarkRunner
│   ├── safety.py              # SafetyEvaluator
│   └── reporting.py           # EvaluationReport
│
├── nslt/
│   ├── __init__.py
│   ├── model.py               # ~694 lines — NSLTModel, MoENSLTModel
│   ├── layer1_ssm.py          # ~151 lines — SSMCompressionEngine
│   ├── layer2_ltc.py          # ~200 lines — LTCRoutingLayer
│   ├── layer3_sandbox.py      # ~300 lines — LatentSandbox, EnergyFunction
│   ├── layer4_output.py       # ~236 lines — SparseOutputSynthesizer
│   ├── ssm_scan.py            # ~445 lines — 5 scan backends + autograd
│   ├── moe_ssm.py             # ~200 lines — MoE SSM blocks
│   ├── mcts_sandbox.py        # ~400 lines — MCTS latent sandbox
│   ├── multiscale_ssm.py      # ~241 lines — 3-level state hierarchy
│   └── vision_encoder.py      # ~200 lines — SigLIP vision encoder
│
└── utils/
    └── reproducibility.py     # set_seed() for deterministic training
```

### 10.2 Key file summaries

| File | Lines | Key classes/functions | Dependencies |
|---|---|---|---|
| `main.py` | ~250 | `cmd_*` dispatchers | All `src.*` modules |
| `src/config/schema.py` | 463 | `Config`, `ModelArchitectureConfig`, `TrainingConfig`, etc. | pydantic, yaml |
| `src/models/factory.py` | 447 | `ModelFactory` | transformers, torch, `src.config.schema` |
| `src/training/pipeline.py` | 365 | `TrainingPipeline` | transformers.Trainer, `src.*` |
| `src/infrastructure/distributed.py` | 108 | `DistributedSetup` | torch.distributed |
| `src/tokenizer_trainer.py` | 222 | `download_tokenizer`, `train_custom_tokenizer`, `ensure_tokenizer` | tokenizers, transformers |
| `src/nslt/model.py` | 694 | `NSLTModel`, `MoENSLTModel`, `TokenEmbedding`, `RotaryPositionEncoding` | torch.nn |
| `src/nslt/ssm_scan.py` | 445 | `selective_scan`, `SSMScanFunction`, `selective_scan_triton` | torch, triton (optional) |
| `src/data/pipeline.py` | 241 | `DataPipeline` | datasets, transformers |
| `src/data/quality.py` | 147 | `QualityFilter`, `ExactDeduplicator`, `ContaminationFilter` | re, hashlib |

---

## 11. Test Suite

109 tests across 11 files.

| File | Tests | Key fixtures | What's tested |
|---|---|---|---|
| `tests/test_nslt.py` | 17 | `nslt_model` | SSM shapes, LTC ODE, sandbox, sparse output, full model forward/train/gen |
| `tests/test_integration.py` | 12 | `tiny_nslt`, `tiny_moe` | SSM scan correctness, loss convergence, MoE forward, vision encoder, MCTS |
| `tests/test_alignment.py` | 4 | — | DPO loss prefers chosen, ORPO loss finite, SimPO correctness |
| `tests/test_config.py` | 7 | — | Loading, defaults, v1→v2 migration, invalid dtype rejection |
| `tests/test_data_collector.py` | 3 | — | Chat schema, problem-solution, max_samples |
| `tests/test_data_pipeline.py` | 20 | — | Text cleaning, language detection, field extraction, quality filtering |
| `tests/test_evaluation.py` | 4 | — | Safety keywords, benchmark result formatting |
| `tests/test_generation.py` | 8 | — | Code extraction, language aliasing, model not-ready error |
| `tests/test_quality.py` | 6 | — | Length check, low-quality markers, dedup, contamination |
| `tests/test_trainer.py` | 3 | — | Constitutional prompt, supervised tokenization, format |
| `tests/test_validation.py` | 10 | — | Python syntax, execution, timeout, batch validation |

**Run all:** `python -m pytest tests/ -q`
**With coverage:** `python -m pytest tests/ --cov=src --cov-report=term-missing`

---

## 12. Benchmarks

| Name | Type | # Problems | Metric | Parameters |
|---|---|---|---|---|
| `human_eval` | Code generation | 164 | pass@1 | 512 max tokens, temp 0.8 |
| `mbpp` | Code generation | 417 | pass@1 | Same as HumanEval |
| `mmlu` | Knowledge (57 subjects) | ~14K | accuracy | 5-shot, answer letter only |
| `hellaswag` | Commonsense NLI | 10K | accuracy | 0-shot, pick correct ending |
| `arc` | Science QA | 2,590 (challenge) | accuracy | 0-shot, multiple choice |
| `gsm8k` | Math word problems | 1,319 | exact match | 8-shot CoT |
| `truthfulqa` | Factuality | 817 | mc1/mc2 | 0-shot, multiple choice |
| `winogrande` | Coreference | 1,267 | accuracy | 0-shot, fill-in-the-blank |
| `bbh` | Reasoning (23 tasks) | 6,511 | accuracy | 3-shot CoT |

All benchmarks use `BenchmarkRunner.run_benchmarks(benchmark_list)`.

---

## 13. Known Flaky Tests & Edge Cases

### Flaky test
- `test_nslt_loss_decreases` (`tests/test_integration.py`): Trains a tiny NSLT
  for 10 steps with random init. Loss may occasionally increase (~1/5 runs).
  **Not a bug** — rerun the test. To stabilize, increase model size or fix seed
  in the test fixture.

### Edge cases
1. **Tokenizer without pad_token:** Auto-fixed by `add_special_tokens({"pad_token": "<pad>"})`
2. **Checkpoint with different vocab_size:** Detected by `is_compatible()`,
   returns `False`, starts fresh
3. **Zero samples from data pipeline:** `_load_stage_data` catches `Exception`,
   logs warning, returns `None` → stage skipped
4. **Tokenizer directory doesn't exist:** `ensure_tokenizer()` downloads/trains
   automatically. If it fails, raises `RuntimeError` with actionable message
5. **Single-GPU FSDP:** FSDP works on single GPU (no sharding). Optimizer states
   are not sharded, so fp32 may OOM on single A100 80GB for the full 8.18B model
6. **Ctrl+C during training:** `try/finally` in `cmd_full_training` ensures
   `dist.destroy_process_group()` is called. Safe to re-run immediately
7. **Multiple torchrun instances:** NCCL will hang if two `torchrun` processes
   share the same `MASTER_ADDR:MASTER_PORT`. Ensure only one instance runs
8. **Vision encoder enabled but no image data:** `VisionConfig.enabled: true`
   with text-only data will cause shape errors. Keep `enabled: false` for
   text-only training
