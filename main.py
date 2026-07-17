from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
import torch

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(_PROJECT_DIR, "hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(_PROJECT_DIR, "hf_cache", "datasets")
os.environ["HF_HUB_CACHE"] = os.path.join(_PROJECT_DIR, "hf_cache", "hub")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ===================================================================
# COMMANDS
# ===================================================================

def cmd_full_training(args: argparse.Namespace) -> None:
    """Run the complete training sequence (pretrain -> SFT -> instruction tuning)."""
    from src.config.schema import load_config
    from src.training.pipeline import TrainingPipeline
    from src.infrastructure.distributed import DistributedSetup
    from src.tokenizer_trainer import ensure_tokenizer

    cfg = load_config(args.config)
    dist = DistributedSetup(cfg)

    if dist.is_main_process():
        ensure_tokenizer(config_path=args.config, force=args.fresh_start)
    if dist.is_distributed:
        import torch.distributed as dist_pkg
        dist_pkg.barrier()

    pipeline = TrainingPipeline(cfg)
    try:
        pipeline.initialize(fresh_start=args.fresh_start)
        results = pipeline.full_training_sequence()
        pipeline.save_model()
    finally:
        pipeline.cleanup()
    logger.info("Training complete!")


def cmd_config_validate(args: argparse.Namespace) -> None:
    """Validate the configuration file."""
    from src.config.schema import Config, load_config
    from src.models.factory import ModelFactory
    from pydantic import ValidationError
    try:
        cfg = load_config(args.config)
        print("Configuration is valid!")
        print(f"  Model: {cfg.model.name}")
        arch = cfg.model.architecture
        print(f"  Architecture: {arch.model_type} {arch.hidden_size}")
        print(f"  Max context: {arch.max_position_embeddings}")
        print(f"  Training stages: pretrain={cfg.training.pretrain.enabled}, sft={cfg.training.sft.enabled}")
        print(f"  Distributed: {cfg.distributed.strategy}")
        print(f"  Datasets: {len(cfg.data.datasets)}")
        estimate = ModelFactory.estimate_model_size(arch)
        print(f"  Estimated params: {estimate['total_params_b']}B")
    except (yaml.YAMLError, ValidationError, FileNotFoundError) as e:
        print(f"Configuration INVALID: {e}")
        sys.exit(1)


def cmd_info(args: argparse.Namespace) -> None:
    """Print system and model information."""
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
    print(f"Python: {sys.version}")


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate code from a prompt using a trained checkpoint."""
    from src.config.schema import load_config
    from src.models.factory import ModelFactory
    from transformers import AutoTokenizer

    cfg = load_config(args.config)
    tokenizer = ModelFactory.load_tokenizer(args.tokenizer)
    model, _ = ModelFactory.load_model(args.checkpoint, cfg, tokenizer)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    prompt = args.prompt or sys.stdin.read().strip()
    if not prompt:
        print("Error: no prompt provided. Use --prompt or pipe text to stdin.")
        sys.exit(1)

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    output_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(generated)


def cmd_download_tokenizer(args: argparse.Namespace) -> None:
    """Download a tokenizer from HuggingFace Hub."""
    from src.config.schema import load_config
    from src.tokenizer_trainer import download_tokenizer
    config_path = os.path.join(_PROJECT_DIR, args.config)
    model_id = args.model_id
    if not model_id:
        try:
            cfg = load_config(config_path)
            model_id = cfg.tokenizer.huggingface_model
            print(f"[*] Using model from config: {model_id}")
        except Exception as e:
            model_id = "Xenova/claude-tokenizer"
            print(f"[!] Config load failed ({e}), falling back to {model_id}")
    else:
        print(f"[*] Using explicit model: {model_id}")
    output_dir = os.path.join(_PROJECT_DIR, args.output)
    print(f"[*] Saving tokenizer to: {output_dir}")
    download_tokenizer(
        output_dir=output_dir,
        model_id=model_id,
        force=args.force,
    )
    print(f"[*] Tokenizer download complete: {output_dir}")


def cmd_test(args: argparse.Namespace) -> None:
    """Run the test suite."""
    import subprocess
    import sys
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    if args.filter:
        cmd.extend(["-k", args.filter])
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    sys.exit(result.returncode)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run coding benchmarks against a trained checkpoint."""
    from src.config.schema import load_config
    from src.evaluation.benchmarks import BenchmarkRunner
    from src.models.factory import ModelFactory

    cfg = load_config(args.config)
    tokenizer = ModelFactory.load_tokenizer(args.tokenizer)
    model, _ = ModelFactory.load_model(args.checkpoint, cfg, tokenizer)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    benchmarks = args.benchmarks.split(",") if args.benchmarks else ["human_eval", "mbpp"]
    runner = BenchmarkRunner(model, tokenizer)
    results = runner.run_benchmarks(benchmarks)
    for r in results:
        print(f"{r.name}: {r.score:.2%}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zmodel", description="Methos Class Model - Neural-State Liquid Transformer")
    parser.add_argument("--config", default="config.yaml")

    sub = parser.add_subparsers(dest="command", required=True)

    p_full = sub.add_parser("full-training", help="Run full training (pretrain -> SFT -> instruction tuning)")
    p_full.add_argument("--fresh-start", action="store_true", help="Ignore existing checkpoints")
    p_full.add_argument("--config", default="config.yaml")

    sub.add_parser("config-validate", help="Validate configuration file")
    sub.add_parser("info", help="Print system information")

    p_gen = sub.add_parser("generate", help="Generate code from a prompt")
    p_gen.add_argument("--prompt", type=str, help="Prompt text (or pipe to stdin)")
    p_gen.add_argument("--checkpoint", type=str, default="models/fable5", help="Model checkpoint path")
    p_gen.add_argument("--tokenizer", type=str, default="models/tokenizer", help="Tokenizer path")
    p_gen.add_argument("--max-new-tokens", type=int, default=1024)
    p_gen.add_argument("--temperature", type=float, default=0.7)
    p_gen.add_argument("--top-k", type=int, default=40)
    p_gen.add_argument("--top-p", type=float, default=0.9)

    p_test = sub.add_parser("test", help="Run the test suite")
    p_test.add_argument("--filter", type=str, help="Filter tests by keyword (-k)")

    p_dl = sub.add_parser("download-tokenizer", help="Download a tokenizer from HuggingFace Hub")
    p_dl.add_argument("--model-id", type=str, default="", help="HuggingFace model ID (default: from config)")
    p_dl.add_argument("--output", type=str, default="models/tokenizer", help="Output directory")
    p_dl.add_argument("--config", default="config.yaml", help="Path to config file")
    p_dl.add_argument("--force", action="store_true", help="Overwrite existing tokenizer")

    p_bench = sub.add_parser("benchmark", help="Run coding benchmarks")
    p_bench.add_argument("--checkpoint", type=str, default="models/fable5", help="Model checkpoint path")
    p_bench.add_argument("--tokenizer", type=str, default="models/tokenizer", help="Tokenizer path")
    p_bench.add_argument("--benchmarks", type=str, default="human_eval,mbpp", help="Comma-separated benchmark names")

    return parser


def main() -> None:
    cmd_map = {
        "full-training": cmd_full_training,
        "config-validate": cmd_config_validate,
        "info": cmd_info,
        "generate": cmd_generate,
        "test": cmd_test,
        "download-tokenizer": cmd_download_tokenizer,
        "benchmark": cmd_benchmark,
    }

    is_distributed = os.environ.get("LOCAL_RANK") is not None
    world_size = int(os.environ.get("WORLD_SIZE", "0"))

    if is_distributed and world_size > 1:
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        if rank == 0:
            print(f"[*] Distributed mode with FSDP ({world_size} GPUs)")
    else:
        import subprocess
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                encoding="utf-8",
            )
            free_mems = [int(x) for x in out.strip().split("\n")]
            best = free_mems.index(max(free_mems))
            if free_mems[best] < 10000:
                print(f"[!] Warning: best GPU only has {free_mems[best]}MB free.")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best)
            print(f"[*] Single-GPU: GPU {best} ({free_mems[best]}MB free)")
        except Exception as e:
            print(f"[!] GPU selection failed: {e}")

    parser = build_parser()
    args = parser.parse_args()

    handler = cmd_map.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
