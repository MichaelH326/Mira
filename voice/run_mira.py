"""
run_mira.py — Talk to Mira locally from a self-contained `mira.mdlo` (v2).

    pip install llama-cpp-python
    python run_mira.py                 # expects mira.mdlo next to this script

What you get:
  - A phone-call-style conversation: short casual spoken replies, full
    history passed back with every turn, /reset and /quit commands.
  - TTS-ready output: replies stream one complete sentence at a time, as
    plain speakable text (the model is prompted to avoid lists, markdown,
    emojis, and digits-as-symbols).
        --tts-cmd 'espeak'      pipe each finished sentence into any command
        --speak                 auto-pick espeak / say if installed
  - Missing-context behavior: Mira is trained and instructed to ask one
    short follow-up question instead of guessing when your message is
    ambiguous.
  - Runs fully offline on a normal laptop CPU. The .mdlo contains the whole
    quantized model; nothing else is downloaded.

(v1 char-level .mdlo files are handled by run_mira_v1.py, kept in this repo.)
"""

import os
import re
import sys
import json
import struct
import shutil
import hashlib
import argparse
import subprocess

MDLO_MAGIC = b"MDLO"


# ------------------------------ container ------------------------------------

def read_mdlo(path):
    with open(path, "rb") as f:
        head = f.read(16)
        if head[:4] != MDLO_MAGIC:
            raise SystemExit(f"{path} is not an MDLO file")
        version = struct.unpack("<I", head[4:8])[0]
        if version == 1:
            raise SystemExit("This is an MDLO v1 (char-level) file — run it with run_mira_v1.py")
        if version != 2:
            raise SystemExit(f"Unsupported MDLO version {version}")
        header_len = struct.unpack("<Q", head[8:16])[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        payload = f.read()

    if hashlib.sha256(payload).hexdigest() != header["payload_sha256"]:
        raise SystemExit("Payload checksum mismatch — the .mdlo file is corrupt")
    return header, payload


def extract_gguf(header, payload):
    """llama.cpp mmaps a file from disk, so unwrap the GGUF into a local cache
    (keyed by content hash — repeat runs reuse it instantly)."""
    cache = os.path.join(os.path.expanduser("~"), ".cache", "mira")
    os.makedirs(cache, exist_ok=True)
    dest = os.path.join(cache, header["payload_sha256"][:16] + ".gguf")
    if not (os.path.exists(dest) and os.path.getsize(dest) == header["payload_bytes"]):
        tmp = dest + ".tmp"
        with open(tmp, "wb") as f:
            f.write(payload)
        os.replace(tmp, dest)
    return dest


# ------------------------------ speech output --------------------------------

SENT_END = re.compile(r'([.!?…]+["\')\]]?\s+)')


def sentence_stream(token_iter):
    """Groups a stream of text chunks into complete sentences (TTS units)."""
    buf = ""
    for chunk in token_iter:
        buf += chunk
        while True:
            m = SENT_END.search(buf)
            if not m:
                break
            yield buf[: m.end()].strip()
            buf = buf[m.end():]
    if buf.strip():
        yield buf.strip()


class Speaker:
    def __init__(self, tts_cmd=None, auto=False):
        self.cmd = tts_cmd
        if auto and not self.cmd:
            for cand in (["espeak"], ["say"], ["espeak-ng"]):
                if shutil.which(cand[0]):
                    self.cmd = " ".join(cand)
                    break
            if not self.cmd:
                print("(no TTS engine found — install espeak, or use --tts-cmd)", file=sys.stderr)

    def say(self, sentence):
        if not self.cmd:
            return
        try:
            subprocess.run(self.cmd, input=sentence.encode(), shell=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        except Exception as e:
            print(f"(tts failed: {e})", file=sys.stderr)
            self.cmd = None


# ------------------------------ conversation ---------------------------------

_ONES = ["zero","one","two","three","four","five","six","seven","eight","nine","ten",
         "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
         "eighteen","nineteen"]
_TENS = ["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]


def _int_to_words(n):
    """Small self-contained integer speller (keeps run_mira dependency-free)."""
    if n < 0:
        return "minus " + _int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + (" " + _int_to_words(rest) if rest else "")
    for div, name in ((1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")):
        if n >= div:
            rest = n % div
            return _int_to_words(n // div) + f" {name}" + (" " + _int_to_words(rest) if rest else "")
    return str(n)


def _say_number(m):
    tok = m.group(0)
    # year-like: 1995 -> nineteen ninety-five
    if re.fullmatch(r"\d{4}", tok) and 1100 <= int(tok) <= 2099 and int(tok) % 1000 != 0:
        hi, lo = int(tok[:2]), int(tok[2:])
        if lo == 0:
            return _int_to_words(hi) + " hundred"
        return f"{_int_to_words(hi)} {_int_to_words(lo) if lo >= 10 else 'oh ' + _int_to_words(lo)}"
    if "." in tok:
        whole, frac = tok.split(".", 1)
        w = _int_to_words(int(whole.replace(",", "") or 0))
        return w + " point " + " ".join(_int_to_words(int(d)) for d in frac)
    try:
        return _int_to_words(int(tok.replace(",", "")))
    except ValueError:
        return tok


def numbers_to_speech(text):
    """Digits are fine on screen but wrong for TTS; say them as words."""
    text = re.sub(r"(\d)\s*-\s*(\d)", r"\1 to \2", text)          # 15-20 -> 15 to 20
    def _money(m):
        whole, cents = m.group(1).replace(",", ""), m.group(2)
        out = _int_to_words(int(whole)) + (" dollar" if whole == "1" else " dollars")
        if cents and int(cents):
            out += " and " + _int_to_words(int(cents)) + " cents"
        return out
    text = re.sub(r"\$\s*([\d,]+)(?:\.(\d{2}))?", _money, text)
    text = re.sub(r"([\d,.]+)\s*%",
                  lambda m: _say_number(re.match(r"[\d,.]+", m.group(1))) + " percent", text)
    text = re.sub(r"(?<![a-zA-Z\d])[\d,]+(?:\.\d+)?(?![a-zA-Z\d])", _say_number, text)
    return text


def clean_for_speech(text):
    """Safety net: strip anything a TTS engine would stumble on."""
    text = re.sub(r"[*_#`>|~\[\]{}]", "", text)          # markdown leftovers
    text = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]", "", text)  # emoji
    text = "".join(c for c in text if c.isprintable() or c.isspace())  # control chars
    text = numbers_to_speech(text)
    return re.sub(r"\s+", " ", text).strip()


def chat_turn(llm, messages, chat_cfg, speaker, show_text=True):
    """One assistant reply, streamed sentence by sentence. Returns full reply."""
    stream = llm.create_chat_completion(
        messages=messages,
        temperature=chat_cfg.get("temperature", 0.7),
        top_p=chat_cfg.get("top_p", 0.9),
        max_tokens=chat_cfg.get("max_reply_tokens", 120),
        stream=True,
    )

    def chunks():
        for part in stream:
            delta = part["choices"][0]["delta"]
            if "content" in delta and delta["content"]:
                yield delta["content"]

    sentences = []
    if show_text:
        print("Mira: ", end="", flush=True)
    for sent in sentence_stream(chunks()):
        sent = clean_for_speech(sent)
        if not sent:
            continue
        sentences.append(sent)
        if show_text:
            print(sent, end=" ", flush=True)
        speaker.say(sent)          # speak while the next sentence generates
    if show_text:
        print()
    return " ".join(sentences)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mira.mdlo")
    p.add_argument("--ctx", type=int, default=2048, help="context window tokens")
    p.add_argument("--threads", type=int, default=None, help="CPU threads (default: all)")
    p.add_argument("--tts-cmd", default=None,
                   help="shell command that reads a sentence on stdin and speaks it, e.g. 'espeak'")
    p.add_argument("--speak", action="store_true", help="auto-detect espeak/say and speak replies")
    p.add_argument("--system", default=None, help="override the persona baked into the .mdlo")
    p.add_argument("--selftest", action="store_true", help="scripted exchange for CI, then exit")
    args = p.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"'{args.model}' not found. Download it from your GitHub Actions "
                         f"artifacts and place it next to this script.")

    header, payload = read_mdlo(args.model)
    gguf_path = extract_gguf(header, payload)
    del payload

    try:
        from llama_cpp import Llama
    except ImportError:
        raise SystemExit("llama-cpp-python is required:  pip install llama-cpp-python")

    prov = header.get("provenance", {})
    print(f"Loading Mira ({prov.get('base_model', '?')}, {prov.get('quantization', '?')}, "
          f"{header['payload_bytes']/1e6:.0f} MB)...")
    llm = Llama(
        model_path=gguf_path,
        n_ctx=args.ctx,
        n_threads=args.threads or os.cpu_count(),
        verbose=False,
    )

    chat_cfg = header.get("chat", {})
    system_prompt = args.system or chat_cfg.get("system_prompt", "You are Mira.")
    speaker = Speaker(tts_cmd=args.tts_cmd, auto=args.speak)
    messages = [{"role": "system", "content": system_prompt}]

    if args.selftest:
        for msg in ["hey mira", "how long should i nap?", "should i buy it?"]:
            messages.append({"role": "user", "content": msg})
            print(f"You: {msg}")
            reply = chat_turn(llm, messages, chat_cfg, speaker)
            assert reply, "empty reply"
            messages.append({"role": "assistant", "content": reply})
        print("[selftest] OK")
        return

    print("On the line with Mira. /reset starts over, /quit hangs up.\n")
    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_msg:
            continue
        if user_msg == "/quit":
            break
        if user_msg == "/reset":
            messages = [{"role": "system", "content": system_prompt}]
            print("(fresh call)")
            continue

        messages.append({"role": "user", "content": user_msg})
        reply = chat_turn(llm, messages, chat_cfg, speaker)
        messages.append({"role": "assistant", "content": reply})

        # keep the transcript inside the context window: drop oldest turns
        while len(messages) > 40:
            del messages[1:3]


if __name__ == "__main__":
    main()
