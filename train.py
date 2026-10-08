#!/usr/bin/env python3
"""
Training entry point. Loads config/train.yaml and calls mlx_lm.lora.

Usage:
  python train.py
  python train.py --config config/train.yaml
  python train.py --iters 500 --learning-rate 5e-5   # override any yaml key
"""

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_CONFIG = Path("config/train.yaml")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args, extra = parser.parse_known_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    data_dir = Path("data/processed")
    for split in ("train.jsonl", "valid.jsonl"):
        if not (data_dir / split).exists():
            sys.exit(
                f"Missing {data_dir / split}. Run:\n"
                "  python scripts/extract_imessage.py\n"
                "  python scripts/extract_slack.py\n"
                "  python scripts/build_dataset.py"
            )

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--config", str(config_path),
    ] + extra

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
