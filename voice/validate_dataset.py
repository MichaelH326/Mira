"""
validate_dataset.py — Quality gate for Mira's training data.

Run this BEFORE fine-tuning. It fails loudly (exit 1) if the dataset would
produce a bad model, so CI stops in seconds instead of after hours of
training.

Usage:
    python voice/validate_dataset.py voice_data.jsonl
    python voice/validate_dataset.py voice_data.jsonl --min-conversations 5000
"""

import re
import sys
import json
import argparse
import collections

FOREIGN = re.compile(
    r"\b(el|la|los|las|una|del|que|por|para|con|como|más|pero|és|und|der|die|"
    r"das|nicht|ist|für|mit|le|les|des|une|est|pour|avec|dans|sur|che|per|non|"
    r"sono|els|amb|aquest)\b", re.I)
MARKDOWN = re.compile(r"[*_`#\[\]{}|]|^\s*\d+[.):-]\s", re.M)
EMOJI = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]")
URL = re.compile(r"https?://|www\.")
BOILERPLATE = re.compile(r"as an ai|as a language model|i'm just an ai", re.I)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--min-conversations", type=int, default=500)
    p.add_argument("--max-foreign-pct", type=float, default=1.0)
    p.add_argument("--max-markdown-pct", type=float, default=1.0)
    p.add_argument("--max-boilerplate-pct", type=float, default=0.5)
    p.add_argument("--min-unique-pct", type=float, default=15.0,
                   help="unique assistant replies as %% of all replies; low means "
                        "the model will memorize templates instead of generalizing")
    p.add_argument("--min-median-words", type=int, default=6)
    p.add_argument("--max-median-words", type=int, default=45)
    args = p.parse_args()

    convs, replies = 0, []
    bad = collections.Counter()
    behaviors = collections.Counter()

    try:
        with open(args.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msgs = json.loads(line)["messages"]
                except (json.JSONDecodeError, KeyError):
                    bad["malformed lines"] += 1
                    continue
                if not msgs or msgs[0].get("role") != "system":
                    bad["missing system prompt"] += 1
                roles = [m.get("role") for m in msgs[1:]]
                if roles != ["user", "assistant"] * (len(roles) // 2):
                    bad["bad turn alternation"] += 1
                convs += 1
                for m in msgs:
                    if m.get("role") != "assistant":
                        continue
                    c = m.get("content", "")
                    replies.append(c)
                    if not c.strip():           bad["empty replies"] += 1
                    if FOREIGN.search(c):       bad["non-english"] += 1
                    if MARKDOWN.search(c):      bad["markdown/lists"] += 1
                    if EMOJI.search(c):         bad["emoji"] += 1
                    if URL.search(c):           bad["urls"] += 1
                    if BOILERPLATE.search(c):   bad["ai boilerplate"] += 1
                    low = c.lower()
                    if "don't know" in low or "no idea" in low or "no clue" in low:
                        behaviors["says I don't know"] += 1
                    if "can't know" in low or "cannot know" in low or "offline" in low:
                        behaviors["says I can't know"] += 1
                    if c.rstrip().endswith("?"):
                        behaviors["asks a follow-up"] += 1
    except FileNotFoundError:
        print(f"FAIL: {args.path} not found")
        return 1

    if not replies:
        print("FAIL: no assistant replies found")
        return 1

    n = len(replies)
    lens = sorted(len(r.split()) for r in replies)
    median = lens[n // 2]
    unique_pct = 100.0 * len(set(replies)) / n

    print(f"Dataset: {args.path}")
    print(f"  conversations      : {convs:,}")
    print(f"  assistant replies  : {n:,}")
    print(f"  unique replies     : {unique_pct:.1f}%")
    print(f"  median reply length: {median} words")
    print("  behaviors present  : " + (", ".join(
        f"{k} ({100*v/n:.1f}%)" for k, v in behaviors.items()) or "NONE"))
    if bad:
        print("  issues:")
        for k, v in bad.most_common():
            print(f"    {k:22s} {v:6,} ({100*v/n:.1f}%)")

    failures = []
    if convs < args.min_conversations:
        failures.append(f"only {convs:,} conversations (need {args.min_conversations:,})")
    if 100.0 * bad["non-english"] / n > args.max_foreign_pct:
        failures.append(f"non-english {100*bad['non-english']/n:.1f}% > {args.max_foreign_pct}%")
    if 100.0 * bad["markdown/lists"] / n > args.max_markdown_pct:
        failures.append(f"markdown/lists {100*bad['markdown/lists']/n:.1f}% > {args.max_markdown_pct}%")
    if 100.0 * bad["ai boilerplate"] / n > args.max_boilerplate_pct:
        failures.append(f"ai boilerplate {100*bad['ai boilerplate']/n:.1f}% > {args.max_boilerplate_pct}%")
    if bad["empty replies"]:
        failures.append(f"{bad['empty replies']} empty replies")
    if unique_pct < args.min_unique_pct:
        failures.append(f"only {unique_pct:.1f}% unique replies (< {args.min_unique_pct}%) — "
                        f"the model would memorize templates")
    if not (args.min_median_words <= median <= args.max_median_words):
        failures.append(f"median reply {median} words outside "
                        f"{args.min_median_words}-{args.max_median_words}")
    for beh in ("says I don't know", "says I can't know", "asks a follow-up"):
        if behaviors[beh] == 0:
            failures.append(f"no examples of '{beh}' — Mira's honesty contract would be lost")

    if failures:
        print("\nFAIL:")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print("\nPASS: dataset looks good for training")
    return 0


if __name__ == "__main__":
    sys.exit(main())
