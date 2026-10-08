#!/usr/bin/env python3
"""
Parse raw Slack message text from the MCP tool output format and convert
to training pair JSONL.

MCP format (newest-first):
  SenderName <email@domain>: message text ... [YYYY-MM-DD HH:MM:SS TZ]

Multi-line messages have the timestamp only on the last line.

Usage:
  python scripts/parse_mcp_slack.py --data scripts/_slack_data.py --out data/raw/slack.jsonl
"""

import re
import json
import argparse
from pathlib import Path

ANDY_NAME = "Andy Huynh"
CONTEXT_TURNS = 6
MIN_REPLY_LEN = 3
MAX_REPLY_LEN = 1024
SYSTEM_TEMPLATE = "You are Andy. You are messaging {contact} in {channel}."

# Matches the start of a message: "Name <email>:" at the beginning of any line
MSG_START = re.compile(r'^([^<\n]+?)\s<[^>@]+@[^>]+>:\s', re.MULTILINE)
# Trailing timestamp to strip
TIMESTAMP_RE = re.compile(r'\s*\[\d{4}-\d{2}-\d{2}[^\]]*\]\s*$')


def parse_channel_text(raw: str, andy_name: str = ANDY_NAME) -> list[tuple[str, str]]:
    """Parse MCP raw text into (sender, text) tuples, oldest first."""
    matches = list(MSG_START.finditer(raw))
    if not matches:
        return []

    messages = []
    for i, m in enumerate(matches):
        sender = m.group(1).strip()
        text_start = m.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        text = raw[text_start:text_end]
        # strip trailing timestamp
        text = TIMESTAMP_RE.sub("", text).strip()
        # collapse internal newlines to spaces
        text = re.sub(r'\s*\n\s*', " ", text).strip()
        if text:
            messages.append((sender, text))

    # MCP returns newest-first; reverse to chronological
    messages.reverse()
    return messages


def is_junk(text: str) -> bool:
    if not text or len(text) < MIN_REPLY_LEN or len(text) > MAX_REPLY_LEN:
        return True
    if re.sub(r"[^\w]", "", text) == "":
        return True
    if re.fullmatch(r"https?://\S+", text.strip()):
        return True
    return False


def make_pairs(
    messages: list[tuple[str, str]],
    contact: str,
    channel: str,
    andy_name: str = ANDY_NAME,
) -> list[dict]:
    pairs = []
    for i, (sender, text) in enumerate(messages):
        if sender != andy_name:
            continue
        if is_junk(text):
            continue
        start = max(0, i - CONTEXT_TURNS)
        ctx_lines = []
        for s, t in messages[start:i]:
            label = "you" if s == andy_name else contact
            ctx_lines.append(f"{label}: {t}")
        if not ctx_lines:
            continue
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_TEMPLATE.format(contact=contact, channel=channel)},
                {"role": "user", "content": "\n".join(ctx_lines)},
                {"role": "assistant", "content": text},
            ]
        })
    return pairs


def build_from_channels(channels: list[dict], andy_name: str = ANDY_NAME) -> list[dict]:
    all_pairs = []
    for ch in channels:
        if "raw" in ch:
            msgs = parse_channel_text(ch["raw"], andy_name=andy_name)
        else:
            msgs = ch["messages"]   # pre-parsed list of (sender, text) tuples
        pairs = make_pairs(msgs, contact=ch["contact"], channel=ch["channel"], andy_name=andy_name)
        all_pairs.extend(pairs)
        print(f"  {ch['contact']} ({ch['channel']}): {len(msgs)} msgs → {len(pairs)} pairs")
    return all_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Python file defining CHANNELS list")
    parser.add_argument("--out", default="data/raw/slack.jsonl")
    args = parser.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("slack_data", args.data)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    print(f"Processing {len(mod.CHANNELS)} channels...")
    pairs = build_from_channels(mod.CHANNELS)
    print(f"Total: {len(pairs)} training pairs")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(pairs)} examples → {out}")


if __name__ == "__main__":
    main()
