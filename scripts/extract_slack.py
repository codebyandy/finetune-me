#!/usr/bin/env python3
"""
Extract Slack conversations and write chat-pair training examples to
data/raw/slack.jsonl.

Requires a Slack user token (xoxp-...) with scopes:
  channels:history, channels:read, groups:history, groups:read,
  im:history, im:read, mpim:history, mpim:read, users:read

Set the token via:
  export SLACK_USER_TOKEN=xoxp-...

Each output record:
  {
    "source": "slack",
    "channel": "<channel name or DM>",
    "contact": "<display name of other person, or channel name>",
    "context": ["them: hi", "you: hey", ...],
    "reply": "<your reply text>"
  }
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

OUTPUT = Path("data/raw/slack.jsonl")
CONTEXT_TURNS = 6
MIN_REPLY_LEN = 3
MAX_REPLY_LEN = 1024
RATE_LIMIT_SLEEP = 1.2   # seconds between paginated API calls


def get_token() -> str:
    token = os.environ.get("SLACK_USER_TOKEN", "")
    if not token:
        raise SystemExit(
            "Set SLACK_USER_TOKEN=xoxp-... before running.\n"
            "Create a Slack app at https://api.slack.com/apps, add user token scopes,\n"
            "install to workspace, and copy the User OAuth Token."
        )
    return token


def slack_get(token: str, method: str, **params) -> dict:
    import urllib.request, urllib.parse
    url = f"https://slack.com/api/{method}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    if not data.get("ok"):
        raise RuntimeError(f"Slack API error on {method}: {data.get('error')}")
    return data


def get_my_user_id(token: str) -> str:
    return slack_get(token, "auth.test")["user_id"]


def get_user_display_name(token: str, user_id: str, cache: dict) -> str:
    if user_id in cache:
        return cache[user_id]
    try:
        info = slack_get(token, "users.info", user=user_id)
        profile = info["user"]["profile"]
        name = profile.get("display_name") or profile.get("real_name") or user_id
    except Exception:
        name = user_id
    cache[user_id] = name
    return name


def list_all_channels(token: str) -> list[dict]:
    """Return all channels (public, private, DMs, group DMs) the authed user is in."""
    channels = []
    for ch_type in ("public_channel,private_channel", "im", "mpim"):
        cursor = None
        while True:
            params = {"types": ch_type, "limit": 200, "exclude_archived": "true"}
            if cursor:
                params["cursor"] = cursor
            data = slack_get(token, "conversations.list", **params)
            channels.extend(data.get("channels", []))
            cursor = data.get("response_metadata", {}).get("next_cursor", "")
            if not cursor:
                break
            time.sleep(RATE_LIMIT_SLEEP)
    return channels


def fetch_history(token: str, channel_id: str) -> list[dict]:
    """Fetch all messages from a channel, oldest first."""
    messages = []
    cursor = None
    while True:
        params = {"channel": channel_id, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = slack_get(token, "conversations.history", **params)
        messages.extend(data.get("messages", []))
        cursor = data.get("response_metadata", {}).get("next_cursor", "")
        if not cursor or not data.get("has_more"):
            break
        time.sleep(RATE_LIMIT_SLEEP)
    messages.sort(key=lambda m: float(m.get("ts", 0)))
    return messages


def normalize_text(text: str, user_cache: dict, my_id: str, token: str) -> str:
    if not text:
        return ""
    # replace <@U123> with display name
    def replace_mention(m):
        uid = m.group(1)
        return f"@{get_user_display_name(token, uid, user_cache)}"
    text = re.sub(r"<@(U[A-Z0-9]+)>", replace_mention, text)
    # strip channel links and other slack markup
    text = re.sub(r"<#[A-Z0-9]+\|([^>]+)>", r"#\1", text)
    text = re.sub(r"<([^|>]+)\|([^>]+)>", r"\2", text)   # named links
    text = re.sub(r"<([^>]+)>", r"\1", text)              # bare URLs
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_junk(text: str) -> bool:
    if not text or len(text) < MIN_REPLY_LEN or len(text) > MAX_REPLY_LEN:
        return True
    clean = re.sub(r"[^\w]", "", text)
    if not clean:
        return True
    if re.fullmatch(r"https?://\S+", text.strip()):
        return True
    return False


def extract(token: str, dry_run: bool = False) -> list[dict]:
    my_id = get_my_user_id(token)
    print(f"Authenticated as user ID: {my_id}")

    user_cache: dict[str, str] = {}
    channels = list_all_channels(token)
    print(f"Found {len(channels)} channels/DMs")

    examples = []
    for ch in channels:
        ch_id = ch["id"]
        ch_name = ch.get("name", "")
        is_im = ch.get("is_im", False)
        is_mpim = ch.get("is_mpim", False)

        if is_im:
            other_id = ch.get("user", "")
            contact = get_user_display_name(token, other_id, user_cache)
            channel_label = "DM"
        elif is_mpim:
            contact = ch_name or ch_id
            channel_label = f"group-dm:{ch_name}"
        else:
            contact = f"#{ch_name}"
            channel_label = f"#{ch_name}"

        try:
            messages = fetch_history(token, ch_id)
        except RuntimeError as e:
            print(f"  skip {channel_label}: {e}")
            continue

        if dry_run and len(examples) >= 5:
            break

        # Build turn list
        turns: list[tuple[str, str]] = []
        for msg in messages:
            if msg.get("subtype"):   # joins, leaves, etc.
                continue
            uid = msg.get("user", "")
            raw_text = msg.get("text", "")
            text = normalize_text(raw_text, user_cache, my_id, token)
            if not text:
                continue
            if uid == my_id:
                role = "you"
            else:
                role = get_user_display_name(token, uid, user_cache)
            turns.append((role, text))

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
                "source": "slack",
                "channel": channel_label,
                "contact": contact,
                "context": ctx,
                "reply": text,
            })

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUTPUT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = get_token()
    print("Fetching Slack data...")
    examples = extract(token, dry_run=args.dry_run)
    print(f"Extracted {len(examples):,} training pairs from Slack")

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
