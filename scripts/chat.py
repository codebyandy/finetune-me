#!/usr/bin/env python3
"""
CLI chat with your fine-tuned style adapter.

Usage:
  python scripts/chat.py
  python scripts/chat.py --adapter adapters --model mlx-community/Qwen3-4B-8bit
  python scripts/chat.py --contact "Sarah" --channel "DM"   # skip prompts
"""

import argparse
import re
import sys

MODEL_DEFAULT = "mlx-community/Qwen3-4B-8bit"
ADAPTER_DEFAULT = "adapters"
MAX_TOKENS = 256
TEMP = 0.7

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def build_system_prompt(your_name: str, contact: str, channel: str) -> str:
    return f"You are {your_name}. You are messaging {contact} in {channel}."


def format_history(history: list[tuple[str, str]], contact: str, your_name: str) -> str:
    lines = []
    for role, text in history:
        label = your_name if role == "assistant" else contact
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--adapter", default=ADAPTER_DEFAULT)
    parser.add_argument("--your-name", default="Andy")
    parser.add_argument("--contact", default="")
    parser.add_argument("--channel", default="")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--temp", type=float, default=TEMP)
    args = parser.parse_args()

    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        sys.exit("mlx-lm not installed. Run: pip install mlx-lm")

    import pathlib
    adapter_path = pathlib.Path(args.adapter)
    if not adapter_path.exists():
        sys.exit(
            f"Adapter not found at {adapter_path}. "
            "Train first with: python train.py"
        )

    print(f"Loading {args.model} + adapter from {adapter_path} ...")
    model, tokenizer = load(args.model, adapter_path=str(adapter_path))
    print("Model loaded.\n")

    contact = args.contact or input("Who are you talking to? ").strip() or "friend"
    channel = args.channel or input("Which channel or context? ").strip() or "DM"

    system = build_system_prompt(args.your_name, contact, channel)
    print(f"\nSystem: {system}")
    print(f"Type messages as {contact}. Ctrl-C or 'quit' to exit.\n")

    history: list[tuple[str, str]] = []

    while True:
        try:
            user_input = input(f"{contact}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        history.append(("user", user_input))

        context_text = format_history(history[:-1], contact, args.your_name)
        current_msg = f"{contact}: {user_input}"
        user_content = f"{context_text}\n{current_msg}".strip() if context_text else current_msg

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=make_sampler(temp=args.temp),
            verbose=False,
        )

        # strip Qwen3 thinking blocks, speaker prefix, and whitespace
        response = THINK_RE.sub("", response).strip()
        prefix = f"{args.your_name}:"
        if response.startswith(prefix):
            response = response[len(prefix):].strip()

        history.append(("assistant", response))
        print(f"{args.your_name}: {response}\n")


if __name__ == "__main__":
    main()
