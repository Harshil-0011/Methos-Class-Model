from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_dir: str | Path = "logs", level: int = logging.INFO) -> None:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_dir / "train.log")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    for noisy in ("httpx", "datasets", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
