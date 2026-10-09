"""Offline tests for the v2 voice pipeline: python tests/test_voice.py

Needs torch, transformers, peft and tokenizers, but no network: the
fine-tuning test builds a tiny random Llama and a locally trained tokenizer
instead of downloading a base model.
"""

import json
import os
import random
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE = os.path.join(ROOT, "voice")
sys.path.insert(0, VOICE)

import make_voice_dataset as mvd  # noqa: E402
import make_open_voice_dataset as mov  # noqa: E402
from eval_mira import SCENARIOS, build_messages, score  # noqa: E402
from run_mira import fit_history  # noqa: E402


def make_tiny_tokenizer(texts):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    from finetune_mira import CHATML_TEMPLATE

    specials = ["<|im_start|>", "<|im_end|>", "<|pad|>"]
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=400, special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>",
                                  pad_token="<|pad|>")
    tok.chat_template = CHATML_TEMPLATE
    return tok


def sample_conversations(n=200, seed=0):
    rng = random.Random(seed)
    return [mvd.build_conversation(rng)["messages"] for _ in range(n)]


def test_loss_only_on_mira_replies():
    from finetune_mira import encode_conversation

    convs = sample_conversations(50)
    tok = make_tiny_tokenizer(m["content"] for c in convs for m in c)
    conv = convs[0]
    ids, labels = encode_conversation(tok, conv, seq_len=2048)
    trained = tok.decode([i for i, l in zip(ids, labels) if l != -100])
    expected = "".join(m["content"] + "<|im_end|>\n" for m in conv if m["role"] == "assistant")
    assert trained == expected, f"trained on:\n{trained!r}\nexpected:\n{expected!r}"
    assert mvd.SYSTEM_PROMPT not in trained

    _, all_labels = encode_conversation(tok, conv, seq_len=2048, loss_on="all")
    assert all(l != -100 for l in all_labels)

    # replies truncated away entirely -> nothing to learn from this example
    assert encode_conversation(tok, conv, seq_len=8) is None


def test_ask_only_when_nothing_explains_it():
    asks = {f["ask"] for f in mvd.CONTEXT_FRAMES} | {a for _, a in mvd.CLARIFY}
    warm, cold = 0, 0
    setups = {f["setup"] for f in mvd.CONTEXT_FRAMES}
    for msgs in sample_conversations(2000):
        assert msgs[0] == {"role": "system", "content": mvd.SYSTEM_PROMPT}
        roles = [m["role"] for m in msgs[1:]]
        assert roles == ["user", "assistant"] * (len(roles) // 2)
        for i, m in enumerate(msgs):
            if m["role"] == "assistant" and m["content"] in asks:
                # system, greeting, greeting reply, vague question, -> the ask
                assert i == 4, f"asked about 'it' after context existed: {msgs}"
                cold += 1
        warm += any(m["content"] in setups for m in msgs)
    assert warm > 100 and cold > 100, (warm, cold)


def test_one_system_prompt_everywhere():
    import package_mdlo
    assert mov.SYSTEM_PROMPT == mvd.SYSTEM_PROMPT
    assert package_mdlo.DEFAULT_SYSTEM_PROMPT == mvd.SYSTEM_PROMPT


def test_open_converter_keeps_coherent_history():
    rng = random.Random(0)
    first = ("what is a good houseplant for a dark apartment",
             "A snake plant is a great choice because it tolerates low light and "
             "needs very little water. It is also very hard to kill.")
    follow = ("is that one safe for cats",
              "Snake plants are mildly toxic to cats, so keep it somewhere your cat "
              "cannot chew on it. A spider plant is a safer pick.")
    unusable = ("write python code for it", "```python\nprint('hi')\n```")

    msgs = mov.convert_pairs([first, follow], rng, max_turns=4, max_sentences=2)
    assert msgs and len(msgs) == 5, "referential follow-up with its context was dropped"

    # the same reference with nothing before it can't be answered
    assert mov.convert_pairs([follow], rng, 4, 2) is None

    # a rejected turn ends the conversation instead of leaving a gap
    msgs = mov.convert_pairs([first, unusable, follow], rng, 4, 2)
    assert msgs and len(msgs) == 3


def test_history_trimming_keeps_newest():
    class FakeLlm:
        def tokenize(self, b, add_bos=False):
            return b.split()

    sys_msg = {"role": "system", "content": "sys"}
    msgs = [sys_msg]
    for i in range(30):
        msgs += [{"role": "user", "content": "word " * 20},
                 {"role": "assistant", "content": "word " * 20}]
    msgs.append({"role": "user", "content": "newest question"})
    fit_history(FakeLlm(), msgs, n_ctx=400, reply_tokens=100)
    total = sum(len(m["content"].split()) + 6 for m in msgs)
    assert total <= 400 - 100 - 32
    assert msgs[0] is sys_msg and msgs[-1]["content"] == "newest question"
    assert [m["role"] for m in msgs[1:]][-3:] == ["user", "assistant", "user"]


def test_eval_scoring():
    by_cat = {}
    for sc in SCENARIOS:
        by_cat.setdefault(sc["cat"], sc)
    assert score(by_cat["clear"], "a spider has eight legs.")[0]
    assert not score(by_cat["clear"], "which spider do you mean?")[0]
    assert score(by_cat["ambiguous"], "plant what, though?")[0]
    assert not score(by_cat["ambiguous"], "nope, you're fine.")[0]
    assert score(by_cat["unknown"], "honestly, i don't know that one.")[0]
    assert not score(by_cat["unknown"], "it was founded in eighteen twelve.")[0]
    assert score(by_cat["cant_know"], "i can't know that, i'm offline.")[0]
    msgs = build_messages("sys", ["a", "b", "c"])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]

    # held out: no scored prompt appears verbatim in the training set
    train_users = {m["content"] for c in sample_conversations(3000) for m in c
                   if m["role"] == "user"}
    leaked = [sc["turns"][-1] for sc in SCENARIOS if sc["turns"][-1] in train_users]
    assert not leaked, leaked


def test_finetune_runs_and_resumes():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    with tempfile.TemporaryDirectory() as td:
        convs = sample_conversations(40)
        data = os.path.join(td, "data.jsonl")
        with open(data, "w") as f:
            for c in convs:
                f.write(json.dumps({"messages": c}) + "\n")
        base = os.path.join(td, "base")
        tok = make_tiny_tokenizer(m["content"] for c in convs for m in c)
        tok.save_pretrained(base)
        torch.manual_seed(0)
        LlamaForCausalLM(LlamaConfig(
            vocab_size=len(tok), hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=2048,
        )).save_pretrained(base)

        out = os.path.join(td, "out")
        cmd = [sys.executable, os.path.join(VOICE, "finetune_mira.py"), "--base", base,
               "--data", data, "--out", out, "--batch_size", "2", "--grad_accum", "1",
               "--seq_len", "1024"]   # a 400-token vocab makes the system prompt very long
        r = subprocess.run(cmd + ["--steps", "2", "--save_every", "1"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "loss on" in r.stdout and "--loss_on assistant" in r.stdout
        r = subprocess.run(cmd + ["--steps", "3", "--resume"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "Resumed from step 1" in r.stdout

        # a spent time budget stops early but still saves a usable model
        out2 = os.path.join(td, "out2")
        cmd2 = [c if c != out else out2 for c in cmd]
        r = subprocess.run(cmd2 + ["--steps", "50", "--max_minutes", "0.0001"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "Time budget" in r.stdout and "at step 1/50" in r.stdout, r.stdout
        assert os.path.exists(os.path.join(out2, "config.json"))
        assert os.path.exists(os.path.join(out, "config.json"))


if __name__ == "__main__":
    # --no-torch runs only the checks that need nothing beyond the stdlib
    no_torch = "--no-torch" in sys.argv
    needs_torch = {"test_loss_only_on_mira_replies", "test_finetune_runs_and_resumes"}
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and not (no_torch and k in needs_torch)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(tests)} voice tests passed")
