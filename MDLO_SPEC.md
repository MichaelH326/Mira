# MDLO Format Specification (v1)

`.mdlo` ("Mira Deployable Local Object") is a single-file, self-contained
container for a trained Mira model. It holds everything an inference program
needs: architecture config, character tokenizer, and raw weight tensors.
It has **no dependency on PyTorch, pickle, or the training environment** —
a loader only needs to read bytes and parse JSON.

## Layout

| Offset | Size | Field | Description |
|---|---|---|---|
| 0 | 4 | magic | ASCII `MDLO` |
| 4 | 4 | version | uint32 little-endian, currently `1` |
| 8 | 8 | header_len | uint64 little-endian, byte length of the JSON header |
| 16 | header_len | header | UTF-8 JSON (see below) |
| 16 + header_len | — | payload | Concatenated raw tensor bytes, little-endian |

## JSON header

```json
{
  "format": "mdlo",
  "model_name": "mira",
  "created": "2026-08-20T12:00:00Z",
  "config": {
    "vocab_size": 63,
    "n_embd": 128,
    "n_head": 4,
    "n_layer": 4,
    "block_size": 256
  },
  "tokenizer": {
    "type": "char",
    "itos": ["\n", " ", "!", "..."]
  },
  "tensors": [
    {
      "name": "token_embedding_table.weight",
      "dtype": "float32",
      "shape": [63, 128],
      "offset": 0,
      "nbytes": 32256
    }
  ],
  "training": {
    "final_train_loss": 0.41,
    "final_val_loss": 0.44,
    "iters": 6000,
    "dataset_chars": 400123
  }
}
```

Notes:

- `tokenizer.itos` is the full index→character list; `stoi` is derived by
  inverting it. Encoding is: unknown characters are dropped.
- Every tensor is stored as **row-major (C-order) little-endian float32**.
- `offset` is relative to the start of the payload section, not the file.
- Tensor `name`s follow the PyTorch `state_dict` naming of the reference
  architecture (below), so a loader can reconstruct the module tree or, as
  `run_mira.py` does, index them directly by name.
- Integrity: the header includes `"payload_sha256"` with the hex digest of
  the payload bytes. Loaders should verify it.

## Reference architecture

Decoder-only transformer, character-level, pre-LayerNorm:

```
tokens -> tok_emb + pos_emb
       -> N x [ x + MHSA(LN(x)) ; x + MLP(LN(x)) ]
       -> LN -> Linear(vocab) -> logits
```

- MHSA: `n_head` heads, each with separate bias-free `key`/`query`/`value`
  projections of size `n_embd / n_head`, causal mask, scaled by
  `n_embd ** -0.5`, concatenated then passed through a bias-ful `proj`.
- MLP: `Linear(n_embd, 4*n_embd) -> ReLU -> Linear(4*n_embd, n_embd)`.
- LayerNorm: eps `1e-5`, elementwise affine.
- Weight layout for `Linear`: shape `(out_features, in_features)`;
  forward is `y = x @ W.T + b`.

## Versioning

Loaders must reject files whose magic isn't `MDLO` or whose version is
greater than what they support. Additive header fields are allowed within
a version; layout changes require a version bump.

## Version 2: GGUF container (large fine-tuned models)

Same outer layout (magic `MDLO`, uint32 version = `2`, uint64 header_len,
JSON header, payload) and the same `payload_sha256` integrity rule, but the
payload is a complete **GGUF** model file and the header describes chat
behavior instead of raw tensors:

```json
{
  "format": "mdlo",
  "model_name": "mira-casual-0.5b",
  "created": "2026-08-24T17:00:00Z",
  "container": "gguf",
  "chat": {
    "system_prompt": "you are mira, a friendly voice on a casual phone call...",
    "n_ctx": 2048,
    "temperature": 0.7,
    "max_reply_tokens": 80
  },
  "payload_sha256": "..."
}
```

Rationale: at 500M+ parameters, quantized GGUF inference (llama.cpp) is the
right CPU runtime; MDLO v2 wraps it so the user experience stays "one .mdlo
file + run_mira.py". Loaders extract the payload to a local cache (keyed by
checksum) and run it with `llama-cpp-python`. v1 files remain fully
supported by the same loader.

# MDLO Format Specification (v2)

Version 2 targets fine-tuned 360M–2B models. Same outer container as v1,
different payload strategy: instead of raw tensors + a custom engine, the
payload is a **complete quantized GGUF file**, and the header carries the
conversational persona. This keeps the single-file promise while getting
llama.cpp's fast CPU inference (a 360M Q4 model streams faster than
reading speed on a normal laptop).

## Layout

| Offset | Size | Field | Description |
|---|---|---|---|
| 0 | 4 | magic | ASCII `MDLO` |
| 4 | 4 | version | uint32 LE, `2` |
| 8 | 8 | header_len | uint64 LE |
| 16 | header_len | header | UTF-8 JSON (below) |
| 16 + header_len | — | payload | A complete GGUF file, byte for byte |

## v2 JSON header

```json
{
  "format": "mdlo",
  "format_version": 2,
  "model_name": "mira",
  "engine": "gguf",
  "payload_sha256": "…",
  "payload_bytes": 123456789,
  "chat": {
    "system_prompt": "You are Mira, a friendly voice assistant…",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_reply_tokens": 120,
    "style": "casual-voice"
  },
  "provenance": {
    "base_model": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "finetune_steps": 300,
    "quantization": "Q4_K_M"
  }
}
```

Notes:

- The tokenizer and chat template live *inside* the GGUF payload (standard
  GGUF metadata), so the file remains fully self-contained.
- Loaders verify `payload_sha256`, then hand the payload to any
  GGUF-capable runtime. `run_mira.py` unwraps it once into
  `~/.cache/mira/<hash>.gguf` (llama.cpp memory-maps files from disk) and
  reuses the cached copy on later runs.
- The persona in `chat.system_prompt` is a default, not a lock — runners
  may let users override it.
- v1 (raw-tensor, char-level) files remain valid; loaders dispatch on the
  version field.
