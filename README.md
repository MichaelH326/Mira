# Mira

Mira is a small language model (SLM) assistant that **trains and runs entirely on a normal laptop CPU** — no GPU, no cloud, no internet connection, no API keys.

Mira's highest priority is **reliable, context-aware answering rather than answering at all costs**. It is built to value being correct over being confident, asking over assuming, and admitting uncertainty over hallucinating:

- It answers directly when it has enough reliable information.
- It asks one short, relevant follow-up question when required context is missing, instead of guessing.
- It says **"I don't know"** for facts outside its knowledge, instead of inventing names, numbers, or events.
- It says **"I can't know that"** for things that need a clock, sensors, or the internet.
- It states its own limitations honestly (it's a very small offline model, and says so).
- Given a context passage, it answers **from the context** — and says "the context doesn't say" when the answer isn't there.

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only wheel

# 1. Generate the instruction dataset that teaches the honesty behaviors
python -m mira.instruct_data --out-dir data

# 2. Train the assistant (~40 min on a 4-core laptop CPU)
python -m mira.train --data data/instruct_train.txt --preset tiny \
    --max-iters 2500 --out checkpoints/mira-assistant

# 3. Chat with Mira — fully offline
python -m mira.assistant --checkpoint checkpoints/mira-assistant

# 4. Measure the honesty behaviors on held-out prompts
python -m mira.eval_behavior --checkpoint checkpoints/mira-assistant
```

Example session:

```
you>  What is the capital of France?
mira> The capital of France is Paris.
you>  What is the capital?
mira> Which country do you mean?
you>  What is the capital of Zorvania?
mira> I don't know the capital of Zorvania. It isn't something I have reliable information about.
you>  What will the weather be like tomorrow?
mira> I can't know that. I run offline with no access to weather data.
```

## How reliability is engineered

A 0.8M-parameter model cannot know much — so Mira is designed around *knowing what it doesn't know*. Three layers work together:

1. **Behavior training** (`mira/instruct_data.py`). The instruction dataset covers seven behaviors: direct answers for a closed fact base, "I don't know" for unknown entities (trained on fabricated names so the pattern generalizes to anything unfamiliar), honest refusals for unknowable questions, clarifying questions when a required slot is missing, honest capability limits, context-grounded QA with explicit "the context doesn't say" cases, and identity smalltalk. Crucially, the *same question templates* appear both with the slot filled (→ answer) and with it missing (→ ask), which teaches the model to discriminate "enough information" from "not enough" — and to skip unnecessary questions when the context is already sufficient.

2. **Greedy decoding** (`mira/assistant.py`). The assistant decodes deterministically by default — no sampling noise on top of memorized facts.

3. **A confidence gate**. The runtime measures the mean per-token log-probability of every reply. Low confidence means the model is off its training distribution, so the reply is replaced with an honest fallback ("I'm not confident I can answer that reliably...") rather than shown as if it were trustworthy. Tune with `--min-confidence`; inspect with `--show-confidence`.

The evaluation (`mira/eval_behavior.py`) scores each behavior on held-out prompts — including real countries deliberately excluded from training, where the only honest response is "I don't know" — and reports a **hallucination rate**: how often Mira produced a confident made-up answer where honesty was required. That is the metric Mira is built to minimize.

## Project layout

```
mira/
├── config.py         # MiraConfig + tiny/small/base presets (0.8M–6.5M params)
├── tokenizer.py      # char-level tokenizer + dialogue marker tokens (<|u|>, <|m|>, <|e|>)
├── model.py          # GPT-style transformer: pre-norm blocks, SDPA attention, weight tying
├── data.py           # corpus loading and random-crop batching
├── train.py          # AdamW + cosine LR schedule, checkpointing, --resume
├── instruct_data.py  # synthetic instruction dataset teaching the honesty behaviors
├── assistant.py      # offline chat runtime: greedy decoding + confidence gate
├── generate.py       # free-form sampling CLI (for plain-text models)
└── eval_behavior.py  # held-out behavior accuracy + hallucination rate
data/
├── instruct_train.txt     # generated training stream
├── instruct_eval.jsonl    # held-out behavior eval prompts
└── tinyshakespeare.txt    # optional corpus for free-form pretraining demos
tests/
└── test_smoke.py     # tokenizer/model/dataset checks (also verifies no eval leakage)
```

## Model presets

| Preset  | Params | Layers | Heads | Embed | Context | CPU training time* |
|---------|--------|--------|-------|-------|---------|--------------------|
| `tiny`  | ~0.8M  | 4      | 4     | 128   | 256     | ~40 min            |
| `small` | ~2.7M  | 6      | 6     | 192   | 256     | ~1.5–3 h           |
| `base`  | ~6.5M  | 8      | 8     | 256   | 512     | an evening         |

\* default iteration counts on a 4-core laptop; scales with cores. `--resume` continues from the last checkpoint, so training can be stopped and picked up any time. `--threads N` caps CPU usage if you want to keep working while it trains.

## Honest limitations

Mira is an educational-scale model, and — true to its own values — this README won't overclaim. At under a million parameters trained on a synthetic dataset, Mira's *knowledge* is a small closed fact base (capitals, colors, opposites, single-digit arithmetic, spelling, definitions, calendar order), and its language understanding is limited to short, simple exchanges. What it demonstrates is the *behavioral* contract: within and around that domain it reliably answers what it knows, asks for what's missing, and declines to invent what it can't know — the failure mode it is engineered against is the confident fabrication. Cross-turn memory is limited to the few most recent exchanges (`--history-turns`).

You can also train the same architecture on plain text (e.g. the included Shakespeare corpus) with `python -m mira.train --data data/tinyshakespeare.txt` and sample it with `python -m mira.generate`.

---

# Mira Voice (v2)

The `voice/` directory scales Mira from the educational from-scratch model
above to a **real conversational model (360M–2B parameters)** with a casual
phone-call personality — short spoken replies, TTS-ready output — that still
runs entirely on a laptop CPU.

Since pretraining at that size on CPU is infeasible (billions of tokens),
v2 **fine-tunes an open Apache-2.0 base** (SmolLM2, Qwen2.5) on a synthetic
phone-call dialogue dataset, then packages the result into a single
self-contained **`.mdlo` v2** file (see `MDLO_SPEC.md`): a quantized GGUF
payload plus persona and settings, verified by checksum.

## Use it

```bash
pip install llama-cpp-python
python voice/run_mira.py                     # mira.mdlo next to your shell
python voice/run_mira.py --speak             # speak replies via espeak/say
python voice/run_mira.py --tts-cmd 'espeak'  # pipe sentences to any TTS
```

Replies stream one finished sentence at a time so a TTS engine can start
speaking while the next sentence generates. Full conversation history is
kept; when your message is ambiguous, Mira asks one short follow-up instead
of guessing — the same honesty contract as v1, carried into v2 via training
data and the baked-in system prompt.

## Build the model

Run the **Train Mira Voice (v2)** workflow from the Actions tab (pick a base
model, steps, and quantization) — it fine-tunes on the CPU runner, packages
`mira.mdlo`, smoke-tests it, uploads it as an artifact, and publishes it as
a GitHub Release. 360M–0.5B bases fit comfortably in one run; for 1.5B+ use
~50–100 steps or fine-tune locally. `steps: 0` skips training and packages
the base with Mira's persona (always fits, any size).

## Bigger, more varied training data

The handcrafted set has ~70 templates — enough to teach the persona, but a
model trained only on it memorizes those replies. `voice/make_open_voice_dataset.py`
scales it up by streaming open instruction corpora (Ai2's Tulu 3, SmolTalk2)
and converting them to spoken register: markdown stripped, trimmed to one or
two sentences, contractions applied, numbers spelled out, and code/math/list/
boilerplate answers rejected outright.

```bash
pip install datasets num2words
python voice/make_open_voice_dataset.py --out voice_data_big.jsonl --n 40000
python voice/finetune_mira.py --data voice_data_big.jsonl --steps 400
```

A quarter of the output (`--core_frac`) is drawn from the handcrafted persona
and abstention set so Mira's identity and honesty behaviors stay strong, and
`--permissive_only` (on by default) skips non-commercially-licensed subsets.
With this dataset you can train considerably longer without memorizing.

## Toward a fully-own pretrained Mira

`voice/make_dolma_voice.py` streams Ai2's open **Dolma** corpus (ODC-By, see
`ATTRIBUTION.md`), keeps only casual spoken-register documents, and rewrites
them TTS-friendly (numbers spelled out, symbols stripped) — a pretraining
corpus for training your own architecture from scratch on a rented GPU,
with the phone-call dialogues as the final phase.
