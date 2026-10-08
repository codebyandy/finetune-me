# local-finetune-experiments

Fine-tune a small language model (Qwen3-4B) to write like you, using your own iMessage and Slack conversations as training data. Runs fully locally on Apple Silicon via [mlx-lm](https://github.com/ml-explore/mlx-lm) LoRA.

The model learns your surface style — sentence rhythm, punctuation, slang, how you open and close messages — and can be conditioned on **who you're talking to** and **which channel**, so it adapts its register accordingly.

> **Privacy note:** Your training data contains other people's messages. Train locally. Don't commit `data/` or publish adapters trained on personal conversations. The `.gitignore` keeps `data/` and `adapters/` out of the repo.

---

## What this is (and isn't)

Fine-tuning picks up style well. It doesn't capture what you know or believe. The model will sound like you while saying plausible nonsense. That's the experiment.

---

## Quickstart

### 1. Install

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and an Apple Silicon Mac with macOS 13.3+.

### 2. Extract your messages

**iMessage** (reads `~/Library/Messages/chat.db` read-only):
```bash
python scripts/extract_imessage.py
# dry run to preview without writing:
python scripts/extract_imessage.py --dry-run
```

**Slack** (uses your configured Slack credentials):
```bash
python scripts/extract_slack.py
# dry run:
python scripts/extract_slack.py --dry-run
```

### 3. Build the training dataset

```bash
python scripts/build_dataset.py
```

Merges both sources, filters noise, formats into chat pairs, and writes `data/processed/{train,valid,test}.jsonl`.

### 4. Train

```bash
python train.py
```

Runs LoRA on `mlx-community/Qwen3-4B-8bit` with settings from `config/train.yaml`. Adapter saved to `adapters/`.

Or call mlx-lm directly:
```bash
mlx_lm.lora --config config/train.yaml
```

### 5. Chat with the fine-tuned model

```bash
python scripts/chat.py
```

Prompts you for who you're talking to and which channel, then runs an interactive session with the adapter.

---

## Per-person conditioning

Each training example includes a system prompt:

```
You are Andy. You are messaging {contact} in {channel}.
```

At inference time (`chat.py`) you supply the same fields. The model has seen this pattern across hundreds of examples per contact, so it learns distinct registers — casual with friends, more structured in work channels.

---

## Training format

Each example is a **chat pair**: the last 6 turns of a conversation as context, followed by your reply as the target. Stored as Qwen3 chat-template JSONL:

```json
{
  "messages": [
    {"role": "system", "content": "You are Andy. You are messaging Sarah in DM."},
    {"role": "user", "content": "Sarah: hey are you coming tonight\nAndy: probably yeah\nSarah: cool what time"},
    {"role": "assistant", "content": "idk maybe 8ish"}
  ]
}
```

---

## Eval ideas

- **Blind quiz:** show a human or LLM judge three replies (real, base model, fine-tuned) and see which one they pick as yours.
- **Perplexity:** compare base model vs. adapter perplexity on held-out test replies.
- **Baseline comparison:** prompt a frontier model with 20 examples of your writing and compare outputs.

---

## Config

`config/train.yaml` controls all training hyperparameters. Key knobs:

| param | default | notes |
|---|---|---|
| `lora_rank` | 16 | higher = more expressive, slower |
| `learning_rate` | 1e-4 | reduce if loss spikes |
| `iters` | 1000 | ~1–2h on M1 for 4B |
| `max_seq_length` | 1024 | covers most conversations |
| `batch_size` | 4 | safe for 16 GB unified memory |

---

## Project structure

```
├── scripts/
│   ├── extract_imessage.py   # pull from ~/Library/Messages/chat.db
│   ├── extract_slack.py      # pull from Slack API
│   ├── build_dataset.py      # clean, merge, format, split
│   └── chat.py               # CLI inference with fine-tuned adapter
├── config/
│   └── train.yaml            # mlx-lm lora config
├── train.py                  # training entry point
├── requirements.txt
└── data/                     # gitignored — never committed
    ├── raw/
    └── processed/
```
