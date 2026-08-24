"""
make_open_voice_dataset.py — Build a large, varied casual-voice dataset for
Mira by converting open instruction datasets into spoken register.

The handcrafted set (make_voice_dataset.py) has only ~70 templates, so a
model trained on it memorizes rather than generalizes. This script streams
big open SFT corpora, keeps the exchanges that can plausibly become casual
speech, and rewrites the assistant turns to sound like a phone call:
markdown stripped, trimmed to one or two spoken sentences, contractions
applied, numbers spelled out, assistant-boilerplate removed.

Sources (all streamed, nothing fully downloaded):
  tulu      allenai/tulu-3-sft-mixture   — Ai2's OLMo/Tulu SFT mixture (ODC-BY)
  tulu2     allenai/tulu-v2-sft-mixture  — older, smaller mixture
  smoltalk  HuggingFaceTB/smoltalk2      — SmolLM3 post-training data

Usage:
    pip install datasets num2words
    python voice/make_open_voice_dataset.py --out voice_data_big.jsonl --n 40000
    python voice/make_open_voice_dataset.py --sources tulu smoltalk --n 100000

By default the handcrafted persona/abstention conversations are mixed in
(--core_frac) so Mira's identity and honesty behaviors stay strong.

LICENSING: Tulu 3 is ODC-BY overall, but bundles subsets under other terms
(e.g. No Robots is CC-BY-NC, GPT4-Alpaca is CC-BY-NC). --permissive_only
(default on) keeps only sources known to be permissively licensed. Keep
ATTRIBUTION.md updated with whatever you use.
"""

import re
import json
import random
import argparse

try:
    from num2words import num2words
    HAVE_N2W = True
except ImportError:
    HAVE_N2W = False

SYSTEM_PROMPT = (
    "You are Mira, a friendly voice assistant on a casual phone call. "
    "Speak in short, natural spoken sentences with contractions, like a relaxed "
    "conversation between friends. Never use lists, markdown, emojis, or symbols; "
    "your words go straight to text to speech. Say numbers the way people say them "
    "out loud. Keep most replies to one or two short sentences. Never guess: if the "
    "question is ambiguous, ask one short follow-up; if you don't know something, say "
    "you don't know; if it needs the internet, a clock, or sensors, say you can't know it."
)

SOURCES = {
    "tulu":     ("allenai/tulu-3-sft-mixture", None),
    "tulu2":    ("allenai/tulu-v2-sft-mixture", None),
    "smoltalk": ("HuggingFaceTB/smoltalk2", "SFT"),
}

# Tulu subsets under permissive terms (no NC clause). Used with --permissive_only.
PERMISSIVE_SOURCES = {
    "flan_v2", "oasst1", "open_assistant_guanaco", "coconot", "wildguardmix",
    "wildjailbreak", "sciriff", "tulu_hard_coded", "personahub", "no_robots_excluded",
}

# ---------------------------------------------------------------------------
# Reject anything that can't become speech
# ---------------------------------------------------------------------------

CODE = re.compile(r"```|\bdef \w+\(|\bclass \w+[:(]|</?\w+>|\bimport \b|;\s*$|\{\s*\"", re.M)
MATH = re.compile(r"\\\(|\\\[|\$\$|\\frac|\\begin\{|\^\{|_\{")
TABLE = re.compile(r"\|.*\|.*\||\+[-=]{3,}\+")
LIST = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", re.M)
BOILERPLATE = re.compile(
    r"as an ai|as a language model|i'm just an ai|i cannot fulfill|"
    r"here (?:is|are) (?:a |an |the )?(?:list|steps|example)|"
    r"step \d+:|firstly,|in conclusion|to summarize|let's break (?:this|it) down",
    re.I)
NON_ASCII = re.compile(r"[^\x00-\x7F]")


def usable(user: str, assistant: str, max_user=220, max_assist=1200) -> bool:
    if not user or not assistant:
        return False
    if not (8 <= len(user) <= max_user):
        return False
    if not (15 <= len(assistant) <= max_assist):
        return False
    both = user + "\n" + assistant
    if CODE.search(both) or MATH.search(both) or TABLE.search(both):
        return False
    if BOILERPLATE.search(assistant):
        return False
    if len(LIST.findall(assistant)) >= 2:          # a real list, not one dash
        return False
    if len(NON_ASCII.findall(both)) / len(both) > 0.02:
        return False
    if user.count("?") == 0 and len(user.split()) > 40:   # long non-question prompt
        return False
    return True


# ---------------------------------------------------------------------------
# Rewrite into casual spoken register
# ---------------------------------------------------------------------------

CONTRACTIONS = [
    (r"\bdo not\b", "don't"), (r"\bdoes not\b", "doesn't"), (r"\bdid not\b", "didn't"),
    (r"\bis not\b", "isn't"), (r"\bare not\b", "aren't"), (r"\bwas not\b", "wasn't"),
    (r"\bwere not\b", "weren't"), (r"\bcannot\b", "can't"), (r"\bcan not\b", "can't"),
    (r"\bwill not\b", "won't"), (r"\bwould not\b", "wouldn't"), (r"\bshould not\b", "shouldn't"),
    (r"\bcould not\b", "couldn't"), (r"\bhave not\b", "haven't"), (r"\bhas not\b", "hasn't"),
    (r"\bit is\b", "it's"), (r"\bthat is\b", "that's"), (r"\bthere is\b", "there's"),
    (r"\byou are\b", "you're"), (r"\bthey are\b", "they're"), (r"\bwe are\b", "we're"),
    (r"\bi am\b", "i'm"), (r"\byou will\b", "you'll"), (r"\bi will\b", "i'll"),
    (r"\byou have\b", "you've"), (r"\blet us\b", "let's"),
]

# stiff -> spoken
DESTIFF = [
    (r"\butilize\b", "use"), (r"\bapproximately\b", "about"), (r"\bnumerous\b", "lots of"),
    (r"\badditionally\b", "also"), (r"\bhowever\b", "but"), (r"\btherefore\b", "so"),
    (r"\bprior to\b", "before"), (r"\bsubsequently\b", "then"), (r"\bassist\b", "help"),
    (r"\brequire\b", "need"), (r"\bpurchase\b", "buy"), (r"\bobtain\b", "get"),
    (r"\bcommence\b", "start"), (r"\bterminate\b", "end"), (r"\bsufficient\b", "enough"),
    (r"\bit is important to note that\b", ""), (r"\bit is worth noting that\b", ""),
    (r"\bplease note that\b", ""), (r"\bin order to\b", "to"),
]

OPENER_FILLERS = ["", "", "", "honestly, ", "yeah, ", "so, ", "oh, ", "well, "]

NUMTOK = re.compile(r"\$[\d,]+(?:\.\d{2})?|[\d.]+%|\b[\d,]+(?:\.\d+)?\b")


def spell_number(tok: str) -> str:
    if not HAVE_N2W:
        return tok
    try:
        m = re.fullmatch(r"\$([\d,]+)(?:\.(\d{2}))?", tok)
        if m:
            out = f"{num2words(int(m.group(1).replace(',', '')))} dollars"
            if m.group(2) and m.group(2) != "00":
                out += f" and {num2words(int(m.group(2)))} cents"
            return out
        m = re.fullmatch(r"([\d.]+)%", tok)
        if m:
            v = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
            return f"{num2words(v)} percent"
        m = re.fullmatch(r"(\d{4})", tok)
        if m and 1500 <= int(m.group(1)) <= 2100:
            return num2words(int(m.group(1)), to="year")
        m = re.fullmatch(r"[\d,]+", tok)
        if m:
            v = int(tok.replace(",", ""))
            if v < 1_000_000_000:
                return num2words(v)
        m = re.fullmatch(r"(\d+)\.(\d+)", tok)
        if m:
            return f"{num2words(int(m.group(1)))} point " + \
                   " ".join(num2words(int(d)) for d in m.group(2))
    except Exception:
        pass
    return tok


def casualize(text: str, rng: random.Random, max_sentences=2) -> str:
    # strip markdown / structure
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", text, flags=re.M)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"[*_`>|~\[\]{}<>#^\\]", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]", "", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2014", ", ")
    text = re.sub(r"\s+", " ", text).strip().lower()

    for pat, rep in DESTIFF:
        text = re.sub(pat, rep, text, flags=re.I)
    for pat, rep in CONTRACTIONS:
        text = re.sub(pat, rep, text, flags=re.I)
    text = NUMTOK.sub(lambda m: spell_number(m.group(0)), text)

    # trim to a spoken-length reply
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    sents = [s for s in sents if 2 <= len(s.split()) <= 40]
    if not sents:
        return ""
    text = " ".join(sents[:max_sentences])

    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text and rng.random() < 0.25:
        text = rng.choice(OPENER_FILLERS) + text
    return text.strip()


def clean_user(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"[*_`>|~\[\]{}<>#^\\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


# ---------------------------------------------------------------------------

def iter_rows(source_key, permissive_only):
    from datasets import load_dataset
    ds_id, subset = SOURCES[source_key]
    ds = load_dataset(ds_id, subset, split="train", streaming=True) if subset \
        else load_dataset(ds_id, split="train", streaming=True)
    for row in ds:
        if permissive_only and source_key.startswith("tulu"):
            src = str(row.get("source") or row.get("dataset") or "")
            if src and not any(p in src for p in PERMISSIVE_SOURCES):
                continue
        msgs = row.get("messages") or []
        pairs = []
        for i in range(len(msgs) - 1):
            if msgs[i].get("role") == "user" and msgs[i + 1].get("role") == "assistant":
                pairs.append((msgs[i].get("content", ""), msgs[i + 1].get("content", "")))
        if pairs:
            yield pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="voice_data_big.jsonl")
    p.add_argument("--sources", nargs="+", default=["tulu"], choices=sorted(SOURCES))
    p.add_argument("--n", type=int, default=40000, help="target conversations to write")
    p.add_argument("--max_turns", type=int, default=4, help="max user/assistant pairs kept per conversation")
    p.add_argument("--max_sentences", type=int, default=2, help="sentences per assistant reply")
    p.add_argument("--core_frac", type=float, default=0.25,
                   help="fraction drawn from the handcrafted persona/abstention set")
    p.add_argument("--permissive_only", action="store_true", default=True,
                   help="skip Tulu subsets with non-commercial licenses")
    p.add_argument("--all_licenses", dest="permissive_only", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    n_core = int(args.n * args.core_frac)
    n_open = args.n - n_core

    written = kept = seen = 0
    with open(args.out, "w", encoding="utf-8") as f:
        # 1) handcrafted persona + abstention conversations
        if n_core:
            from make_voice_dataset import build_conversation
            for _ in range(n_core):
                f.write(json.dumps(build_conversation(rng)) + "\n")
                written += 1
            print(f"Wrote {written:,} handcrafted persona/abstention conversations")

        # 2) converted open-dataset conversations
        for source_key in args.sources:
            print(f"Streaming {SOURCES[source_key][0]} ...", flush=True)
            per_source = n_open // len(args.sources)
            got = 0
            try:
                for pairs in iter_rows(source_key, args.permissive_only):
                    seen += 1
                    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
                    for u, a in pairs[: args.max_turns]:
                        if not usable(u, a):
                            continue
                        cu, ca = clean_user(u), casualize(a, rng, args.max_sentences)
                        if len(ca.split()) < 3 or len(cu.split()) < 2:
                            continue
                        msgs.append({"role": "user", "content": cu})
                        msgs.append({"role": "assistant", "content": ca})
                    if len(msgs) >= 3:
                        f.write(json.dumps({"messages": msgs}) + "\n")
                        written += 1
                        got += 1
                        kept += 1
                    if seen % 20000 == 0:
                        print(f"  seen {seen:,} | kept {kept:,} ({100*kept/max(seen,1):.1f}%)", flush=True)
                    if got >= per_source:
                        break
            except Exception as e:
                print(f"  !! {source_key} failed: {e}")
                print("     (check the dataset id/subset, or your network)")

    print(f"\nWrote {written:,} conversations to {args.out}")
    print("Attribution: Tulu/Dolma data is ODC-BY (Ai2) — keep ATTRIBUTION.md updated.")


if __name__ == "__main__":
    main()
