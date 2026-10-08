#!/usr/bin/env python3
"""
Extract iMessage conversations from ~/Library/Messages/chat.db and write
chat-pair training examples to data/raw/imessage.jsonl.

Each record:
  {
    "source": "imessage",
    "channel": "dm" | "group:<chat_guid>",
    "contact": "<display name or phone>",
    "context": ["them: hi", "you: hey", ...],   # last N turns before reply
    "reply": "<your reply text>"
  }
"""

import argparse
import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

CHAT_DB = Path.home() / "Library" / "Messages" / "chat.db"
OUTPUT = Path("data/raw/imessage.jsonl")
CONTEXT_TURNS = 6       # how many prior messages to include as context
MIN_REPLY_LEN = 3       # drop replies shorter than this
MAX_REPLY_LEN = 1024    # characters


def apple_timestamp_to_dt(ts: int) -> datetime:
    # Apple epoch: 2001-01-01 00:00:00 UTC
    APPLE_EPOCH_OFFSET = 978307200
    return datetime.utcfromtimestamp(ts / 1e9 + APPLE_EPOCH_OFFSET)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    # strip object replacement chars that iMessage uses for attachments
    text = text.replace("￼", "").strip()
    # collapse runs of whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def is_junk(text: str) -> bool:
    if not text or len(text) < MIN_REPLY_LEN or len(text) > MAX_REPLY_LEN:
        return True
    clean = re.sub(r"[^\w]", "", text)
    if not clean:          # pure punctuation / emoji
        return True
    # link-only check
    if re.fullmatch(r"https?://\S+", text.strip()):
        return True
    return False


def resolve_contact(handle_id: str) -> str:
    """Best-effort: return handle_id (phone/email) as-is; strip country code for display."""
    if handle_id.startswith("+1") and len(handle_id) == 12:
        digits = handle_id[2:]
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return handle_id


def extract(db_path: Path) -> list[dict]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Load all chats
    cur.execute("SELECT ROWID, guid, chat_identifier, display_name FROM chat")
    chats = {r["ROWID"]: dict(r) for r in cur.fetchall()}

    # Load all handles (phone/email strings)
    cur.execute("SELECT ROWID, id FROM handle")
    handles = {r["ROWID"]: r["id"] for r in cur.fetchall()}

    # Load all messages joined with their chat
    cur.execute("""
        SELECT
            m.ROWID       AS msg_id,
            m.text        AS text,
            m.is_from_me  AS is_from_me,
            m.date        AS date,
            m.handle_id   AS handle_id,
            cmj.chat_id   AS chat_id
        FROM message m
        JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
        ORDER BY cmj.chat_id, m.date
    """)
    rows = cur.fetchall()
    con.close()

    # Group messages by chat_id
    by_chat: dict[int, list[dict]] = {}
    for r in rows:
        cid = r["chat_id"]
        by_chat.setdefault(cid, []).append(dict(r))

    examples = []
    for chat_id, messages in by_chat.items():
        chat_meta = chats.get(chat_id, {})
        chat_guid = chat_meta.get("guid", "")
        chat_identifier = chat_meta.get("chat_identifier", "")
        display_name = chat_meta.get("display_name") or ""

        # Determine channel and contact strings
        is_group = chat_identifier.startswith("chat")
        if is_group:
            channel = f"group:{chat_guid}"
            contact = display_name if display_name else f"group:{chat_id}"
        else:
            channel = "dm"
            contact = resolve_contact(chat_identifier)

        # Build a flat list of (role, text) in order
        turns: list[tuple[str, str]] = []
        for msg in messages:
            text = normalize_text(msg["text"] or "")
            if not text:
                continue
            role = "you" if msg["is_from_me"] else contact
            turns.append((role, text))

        # Slide a window: whenever "you" sends, use previous CONTEXT_TURNS as context
        for i, (role, text) in enumerate(turns):
            if role != "you":
                continue
            if is_junk(text):
                continue
            start = max(0, i - CONTEXT_TURNS)
            ctx = [f"{r}: {t}" for r, t in turns[start:i]]
            if not ctx:
                continue
            examples.append({
                "source": "imessage",
                "channel": channel,
                "contact": contact,
                "context": ctx,
                "reply": text,
            })

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(CHAT_DB))
    parser.add_argument("--out", default=str(OUTPUT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[error] chat.db not found at {db_path}")
        print("Grant Terminal full disk access in System Settings → Privacy & Security.")
        return

    print(f"Reading {db_path} ...")
    examples = extract(db_path)
    print(f"Extracted {len(examples):,} training pairs from iMessage")

    if args.dry_run:
        for ex in examples[:5]:
            print(json.dumps(ex, indent=2))
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples):,} examples → {out_path}")


if __name__ == "__main__":
    main()
