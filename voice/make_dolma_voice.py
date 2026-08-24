"""
make_dolma_voice.py — Stream OLMo's Dolma corpus, keep only casual
spoken-register documents, and rewrite them to be TTS-friendly.

Two input modes:
  1. Hugging Face streaming (no full download):
       python make_dolma_voice.py --hf allenai/dolma3_mix-6T --target_gb 20
  2. Local Dolma shards (the wget URL-list download path):
       python make_dolma_voice.py --input 'dolma_shards/*.json.gz' --target_gb 20

Output: sharded plain-text files in --outdir, ready for pretraining.

License note: Dolma is ODC-By — keep an ATTRIBUTION file citing Ai2/Dolma
with your dataset. The model you train on it is yours.

Deps: pip install num2words zstandard datasets  (datasets only for --hf mode)
"""

import os
import re
import sys
import glob
import gzip
import json
import argparse

try:
    from num2words import num2words
    HAVE_N2W = True
except ImportError:
    HAVE_N2W = False

# ---------------------------------------------------------------------------
# 1) FILTER — keep only documents that read like casual spoken English
# ---------------------------------------------------------------------------

CONTRACTIONS = re.compile(r"\b\w+'(s|t|re|ve|ll|d|m)\b", re.I)
PRONOUNS = re.compile(r"\b(i|you|we|me|my|your|our|us)\b", re.I)
CODE_HINTS = re.compile(r"(```|\bdef |\bclass |[{};]{2}|</?\w+>|\breturn\b|=>)")
URLS = re.compile(r"https?://\S+|www\.\S+")

def spokenness_score(text: str) -> float:
    """0..1-ish score of how much a document reads like casual speech."""
    n = len(text)
    if n < 400 or n > 40_000:
        return 0.0
    words = text.split()
    if not (60 <= len(words) <= 8_000):
        return 0.0

    # hard rejects: code, markup, heavy symbols/digits, non-ascii soup
    if CODE_HINTS.search(text):
        return 0.0
    sym = sum(text.count(c) for c in "{}[]|<>#*_=~^\\")
    if sym / n > 0.004:
        return 0.0
    digits = sum(c.isdigit() for c in text)
    if digits / n > 0.07:
        return 0.0
    non_ascii = sum(ord(c) > 127 for c in text)
    if non_ascii / n > 0.05:
        return 0.0
    if len(URLS.findall(text)) > 3:
        return 0.0

    # positive signals of casual spoken register
    per_1k = 1000.0 / max(len(words), 1)
    contr = len(CONTRACTIONS.findall(text)) * per_1k          # ~15+/1k in speech
    pron = len(PRONOUNS.findall(text)) * per_1k               # ~60+/1k in speech
    quest = text.count("?") * per_1k * 10
    sentences = re.split(r"[.!?]+", text)
    slens = [len(s.split()) for s in sentences if s.split()]
    avg_slen = sum(slens) / max(len(slens), 1)
    short_bonus = max(0.0, (22 - avg_slen) / 22)              # speech ~10-16 w/sent
    avg_wlen = sum(len(w) for w in words) / len(words)
    plain_bonus = max(0.0, (5.4 - avg_wlen) / 5.4)            # speech uses short words

    score = (min(contr / 15, 1.0) * 0.30 +
             min(pron / 60, 1.0) * 0.30 +
             min(quest / 8, 1.0) * 0.10 +
             short_bonus * 0.15 +
             plain_bonus * 0.15)
    return score


# ---------------------------------------------------------------------------
# 2) TRANSFORM — rewrite kept documents into TTS-safe speakable text
# ---------------------------------------------------------------------------

ABBREV = {
    r"\bdr\.": "doctor", r"\bmr\.": "mister", r"\bmrs\.": "missus",
    r"\bst\.": "street", r"\bvs\.?\b": "versus", r"\betc\.": "and so on",
    r"\be\.g\.": "for example", r"\bi\.e\.": "that is", r"\bapprox\.": "about",
    r"\bhrs?\b": "hours", r"\bmins?\b": "minutes",
}

def spell_number(tok: str) -> str:
    if not HAVE_N2W:
        return tok
    try:
        m = re.fullmatch(r"\$([\d,]+)(?:\.(\d{2}))?", tok)          # $5.99
        if m:
            dollars = num2words(int(m.group(1).replace(",", "")))
            out = f"{dollars} dollar" + ("s" if m.group(1) != "1" else "")
            if m.group(2) and m.group(2) != "00":
                out += f" and {num2words(int(m.group(2)))} cents"
            return out
        m = re.fullmatch(r"([\d.]+)%", tok)                          # 20%
        if m:
            return f"{num2words(float(m.group(1)) if '.' in m.group(1) else int(m.group(1)))} percent"
        m = re.fullmatch(r"(\d{4})", tok)                            # years read as years
        if m and 1500 <= int(m.group(1)) <= 2100:
            return num2words(int(m.group(1)), to="year")
        m = re.fullmatch(r"[\d,]+", tok)                             # plain integers
        if m:
            v = int(tok.replace(",", ""))
            return num2words(v) if v < 1_000_000_000 else tok
        m = re.fullmatch(r"(\d+)\.(\d+)", tok)                       # decimals
        if m:
            return f"{num2words(int(m.group(1)))} point {' '.join(num2words(int(d)) for d in m.group(2))}"
    except Exception:
        pass
    return tok

NUMTOK = re.compile(r"\$[\d,]+(?:\.\d{2})?|[\d.]+%|[\d,]+(?:\.\d+)?")

def make_speakable(text: str) -> str:
    text = URLS.sub(" ", text)
    text = re.sub(r"\S+@\S+", " ", text)                             # emails
    text = re.sub(r"[*_#`>|~\[\]{}<>=+\\^]", " ", text)              # markup/symbols
    text = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]", "", text) # emoji
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("&", " and ").replace("%", " percent") if not HAVE_N2W else text
    for pat, rep in ABBREV.items():
        text = re.sub(pat, rep, text, flags=re.I)
    text = NUMTOK.sub(lambda m: spell_number(m.group(0)), text)
    text = "".join(c for c in text if c.isprintable() or c in "\n ")
    # keep sentences that survived cleanly; drop fragments with leftover junk
    kept = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        s = re.sub(r"\s+", " ", sent).strip()
        if 2 <= len(s.split()) <= 60 and not re.search(r"\d", s):
            kept.append(s)
    return " ".join(kept)


# ---------------------------------------------------------------------------
# 3) STREAM — read docs, score, transform, shard to disk until target size
# ---------------------------------------------------------------------------

def iter_local(pattern):
    for path in sorted(glob.glob(pattern)):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    yield json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue

def iter_hf(dataset_id, subset):
    from datasets import load_dataset
    ds = load_dataset(dataset_id, subset, split="train", streaming=True) if subset \
        else load_dataset(dataset_id, split="train", streaming=True)
    for row in ds:
        yield row.get("text", "")

def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", help="HF dataset id to stream, e.g. allenai/dolma3_mix-6T")
    src.add_argument("--input", help="glob of local dolma shards, e.g. 'shards/*.json.gz'")
    p.add_argument("--subset", default=None, help="HF config/subset name if the dataset has one")
    p.add_argument("--outdir", default="voice_corpus")
    p.add_argument("--target_gb", type=float, default=20.0, help="stop after this much kept text")
    p.add_argument("--min_score", type=float, default=0.45, help="spokenness threshold (0-1)")
    p.add_argument("--shard_mb", type=int, default=256)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    docs = iter_hf(args.hf, args.subset) if args.hf else iter_local(args.input)

    target = int(args.target_gb * 1e9)
    shard_cap = args.shard_mb * 1_000_000
    kept_bytes = seen = kept = shard_idx = 0
    out = open(os.path.join(args.outdir, f"shard_{shard_idx:05d}.txt"), "w", encoding="utf-8")
    shard_bytes = 0

    for text in docs:
        seen += 1
        if spokenness_score(text) >= args.min_score:
            clean = make_speakable(text)
            if len(clean) >= 300:
                out.write(clean + "\n\n")
                b = len(clean.encode()) + 2
                kept += 1
                kept_bytes += b
                shard_bytes += b
                if shard_bytes >= shard_cap:
                    out.close()
                    shard_idx += 1
                    shard_bytes = 0
                    out = open(os.path.join(args.outdir, f"shard_{shard_idx:05d}.txt"), "w", encoding="utf-8")
        if seen % 20_000 == 0:
            rate = 100.0 * kept / seen
            print(f"seen {seen:,} docs | kept {kept:,} ({rate:.1f}%) | "
                  f"{kept_bytes/1e9:.2f}/{args.target_gb} GB", flush=True)
        if kept_bytes >= target:
            break

    out.close()
    print(f"\nDone: {kept:,}/{seen:,} docs kept, {kept_bytes/1e9:.2f} GB "
          f"(~{kept_bytes/4e9:.1f}B tokens) in {args.outdir}/")
    print("Remember: Dolma is ODC-By — ship an ATTRIBUTION file with your dataset.")

if __name__ == "__main__":
    main()
