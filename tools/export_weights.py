#!/usr/bin/env python3
"""Export OpenAI Whisper weights to the WBRF format used by whisper-boring.

Usage:
    python tools/export_weights.py --model tiny --out weights/tiny.wbrf

Requirements:
    pip install openai-whisper torch

Model sizes: tiny, base, small, medium
"""

import sys
import struct
import argparse
import torch
import whisper


# Maps WBRF tensor names to the *openai-whisper* package's own state_dict keys
# (model/model.py's `Whisper` module) — NOT HuggingFace transformers'
# `WhisperModel` naming, which uses a different convention
# (`model.encoder.layers.N.self_attn.q_proj.weight` etc.) that this script's
# earlier version mistakenly used, causing every tensor lookup to fail.
# N is the layer index (0-based), substituted at export time.
ENCODER_FIXED = {
    "encoder.conv1.weight": "encoder.conv1.weight",
    "encoder.conv1.bias":   "encoder.conv1.bias",
    "encoder.conv2.weight": "encoder.conv2.weight",
    "encoder.conv2.bias":   "encoder.conv2.bias",
    "encoder.pos_emb":      "encoder.positional_embedding",
    "encoder.ln.weight":    "encoder.ln_post.weight",
    "encoder.ln.bias":      "encoder.ln_post.bias",
}

ENCODER_BLOCK = {
    "encoder.blocks.{i}.ln1.weight":    "encoder.blocks.{i}.attn_ln.weight",
    "encoder.blocks.{i}.ln1.bias":      "encoder.blocks.{i}.attn_ln.bias",
    "encoder.blocks.{i}.attn_q.weight": "encoder.blocks.{i}.attn.query.weight",
    "encoder.blocks.{i}.attn_q.bias":   "encoder.blocks.{i}.attn.query.bias",
    "encoder.blocks.{i}.attn_k.weight": "encoder.blocks.{i}.attn.key.weight",
    "encoder.blocks.{i}.attn_v.weight": "encoder.blocks.{i}.attn.value.weight",
    "encoder.blocks.{i}.attn_v.bias":   "encoder.blocks.{i}.attn.value.bias",
    "encoder.blocks.{i}.attn_o.weight": "encoder.blocks.{i}.attn.out.weight",
    "encoder.blocks.{i}.attn_o.bias":   "encoder.blocks.{i}.attn.out.bias",
    "encoder.blocks.{i}.ln2.weight":    "encoder.blocks.{i}.mlp_ln.weight",
    "encoder.blocks.{i}.ln2.bias":      "encoder.blocks.{i}.mlp_ln.bias",
    "encoder.blocks.{i}.mlp_fc.weight": "encoder.blocks.{i}.mlp.0.weight",
    "encoder.blocks.{i}.mlp_fc.bias":   "encoder.blocks.{i}.mlp.0.bias",
    "encoder.blocks.{i}.mlp_pr.weight": "encoder.blocks.{i}.mlp.2.weight",
    "encoder.blocks.{i}.mlp_pr.bias":   "encoder.blocks.{i}.mlp.2.bias",
}

DECODER_FIXED = {
    "decoder.tok_emb":   "decoder.token_embedding.weight",
    "decoder.pos_emb":   "decoder.positional_embedding",
    "decoder.ln.weight": "decoder.ln.weight",
    "decoder.ln.bias":   "decoder.ln.bias",
}

DECODER_BLOCK = {
    "decoder.blocks.{i}.ln1.weight":    "decoder.blocks.{i}.attn_ln.weight",
    "decoder.blocks.{i}.ln1.bias":      "decoder.blocks.{i}.attn_ln.bias",
    "decoder.blocks.{i}.sa_q.weight":   "decoder.blocks.{i}.attn.query.weight",
    "decoder.blocks.{i}.sa_q.bias":     "decoder.blocks.{i}.attn.query.bias",
    "decoder.blocks.{i}.sa_k.weight":   "decoder.blocks.{i}.attn.key.weight",
    "decoder.blocks.{i}.sa_v.weight":   "decoder.blocks.{i}.attn.value.weight",
    "decoder.blocks.{i}.sa_v.bias":     "decoder.blocks.{i}.attn.value.bias",
    "decoder.blocks.{i}.sa_o.weight":   "decoder.blocks.{i}.attn.out.weight",
    "decoder.blocks.{i}.sa_o.bias":     "decoder.blocks.{i}.attn.out.bias",
    "decoder.blocks.{i}.ln2.weight":    "decoder.blocks.{i}.cross_attn_ln.weight",
    "decoder.blocks.{i}.ln2.bias":      "decoder.blocks.{i}.cross_attn_ln.bias",
    "decoder.blocks.{i}.ca_q.weight":   "decoder.blocks.{i}.cross_attn.query.weight",
    "decoder.blocks.{i}.ca_q.bias":     "decoder.blocks.{i}.cross_attn.query.bias",
    "decoder.blocks.{i}.ca_k.weight":   "decoder.blocks.{i}.cross_attn.key.weight",
    "decoder.blocks.{i}.ca_v.weight":   "decoder.blocks.{i}.cross_attn.value.weight",
    "decoder.blocks.{i}.ca_v.bias":     "decoder.blocks.{i}.cross_attn.value.bias",
    "decoder.blocks.{i}.ca_o.weight":   "decoder.blocks.{i}.cross_attn.out.weight",
    "decoder.blocks.{i}.ca_o.bias":     "decoder.blocks.{i}.cross_attn.out.bias",
    "decoder.blocks.{i}.ln3.weight":    "decoder.blocks.{i}.mlp_ln.weight",
    "decoder.blocks.{i}.ln3.bias":      "decoder.blocks.{i}.mlp_ln.bias",
    "decoder.blocks.{i}.mlp_fc.weight": "decoder.blocks.{i}.mlp.0.weight",
    "decoder.blocks.{i}.mlp_fc.bias":   "decoder.blocks.{i}.mlp.0.bias",
    "decoder.blocks.{i}.mlp_pr.weight": "decoder.blocks.{i}.mlp.2.weight",
    "decoder.blocks.{i}.mlp_pr.bias":   "decoder.blocks.{i}.mlp.2.bias",
}

N_AUDIO_LAYERS = {"tiny": 4, "base": 6, "small": 12, "medium": 24}
N_TEXT_LAYERS  = {"tiny": 4, "base": 6, "small": 12, "medium": 24}


def build_name_map(model_name: str):
    n_enc = N_AUDIO_LAYERS[model_name]
    n_dec = N_TEXT_LAYERS[model_name]
    mapping = {}  # wbrf_name -> torch_key
    mapping.update(ENCODER_FIXED)
    for i in range(n_enc):
        for wbrf_tmpl, pt_tmpl in ENCODER_BLOCK.items():
            mapping[wbrf_tmpl.format(i=i)] = pt_tmpl.format(i=i)
    mapping.update(DECODER_FIXED)
    for i in range(n_dec):
        for wbrf_tmpl, pt_tmpl in DECODER_BLOCK.items():
            mapping[wbrf_tmpl.format(i=i)] = pt_tmpl.format(i=i)
    return mapping


def write_tensor(f, name: str, tensor: torch.Tensor):
    data = tensor.float().contiguous().cpu().numpy().tobytes()
    n_elems = tensor.numel()
    name_bytes = name.encode("utf-8")
    f.write(struct.pack("<I", len(name_bytes)))
    f.write(name_bytes)
    f.write(struct.pack("<I", n_elems))
    f.write(data)


def export(model_name: str, out_path: str):
    print(f"Loading Whisper {model_name}...")
    model = whisper.load_model(model_name)
    sd = model.state_dict()

    name_map = build_name_map(model_name)
    missing = [wn for wn, pn in name_map.items() if pn not in sd]
    if missing:
        print("WARNING: missing tensors in state dict:", missing[:5], "...", file=sys.stderr)

    tensors = [(wn, sd[pn]) for wn, pn in name_map.items() if pn in sd]
    print(f"Exporting {len(tensors)} tensors to {out_path}...")

    with open(out_path, "wb") as f:
        f.write(b"WBRF")
        f.write(struct.pack("<I", len(tensors)))
        for wbrf_name, tensor in tensors:
            write_tensor(f, wbrf_name, tensor)

    total_bytes = sum(t.numel() * 4 for _, t in tensors)
    print(f"Done. {total_bytes / 1e6:.1f} MB written.")


# ── Tokenizer export ──────────────────────────────────────────────────────────
#
# Current openai-whisper releases use a tiktoken `Encoding` internally, not the
# classic HuggingFace-tokenizers BPE object (`.tokenizer.get_vocab()` /
# `.bpe_ranks` no longer exist). tiktoken exposes only a byte-sequence -> rank
# table (`_mergeable_ranks`), not an ordered merge list, so the classic
# vocab.json / merges.txt pair whisper-boring's tokenizer.br expects has to be
# reconstructed:
#   - vocab.json: {token_text: id, ...} for every rank, id == rank.
#   - merges.txt: one "A B" pair per line, in increasing rank order, recovered
#     by re-running BPE merge selection on each token's raw bytes capped below
#     its own rank (the standard tiktoken -> GPT-2-BPE conversion algorithm;
#     see e.g. huggingface/transformers' TikTokenConverter).
#
# whisper-boring's tokenizer.br concatenates vocab/reverse strings directly
# (it does not implement GPT-2's byte-level unicode remapping), so token text
# is written here as plain UTF-8 (decoded with errors="replace") rather than
# the placeholder-character byte-level representation. This is exact for any
# token that is valid UTF-8 on its own — true for the vast majority of BPE
# pieces — and only degrades (to U+FFFD) for the rare token whose byte
# sequence is a partial multi-byte UTF-8 character. `merges.txt` inherits the
# same simplification; since transcribe() only ever calls `decode()`, not
# `encode()`, this does not affect transcription.

def _bpe_merge_parts(mergeable_ranks, token: bytes, max_rank: int):
    """Re-run BPE merge selection on `token`'s raw bytes, using only merges
    ranked below `max_rank`, to recover the single pair of parts whose
    concatenation is `token` — i.e. the merge rule that created it."""
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i in range(len(parts) - 1):
            rank = mergeable_ranks.get(parts[i] + parts[i + 1])
            if rank is not None and rank < max_rank and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None:
            break
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts


def _recover_merges(mergeable_ranks):
    merges = []
    for token, rank in mergeable_ranks.items():
        if len(token) <= 1:
            continue  # base byte token (or the one reserved empty-byte slot), not a merge
        parts = _bpe_merge_parts(mergeable_ranks, token, max_rank=rank)
        assert len(parts) == 2, f"expected 2 parts for {token!r}, got {parts!r}"
        merges.append((parts[0], parts[1], rank))
    merges.sort(key=lambda m: m[2])
    return merges


def _tok_text(b: bytes) -> str:
    return b.decode("utf-8", errors="replace")


def export_tokenizer(model_name: str, out_dir: str):
    """Export vocab.json and merges.txt from the Whisper (tiktoken) tokenizer."""
    import os
    import json
    import whisper.tokenizer as wt

    os.makedirs(out_dir, exist_ok=True)

    tok = wt.get_tokenizer(multilingual=(model_name != "tiny.en"), language="en", task="transcribe")
    enc = tok.encoding
    mergeable_ranks = enc._mergeable_ranks

    vocab = {_tok_text(b): rank for b, rank in mergeable_ranks.items()}
    vocab.update(tok.special_tokens)  # e.g. "<|endoftext|>", "<|en|>", ...

    vocab_path = os.path.join(out_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"Wrote {len(vocab)} vocab entries to {vocab_path}")

    merges = _recover_merges(mergeable_ranks)
    merges_path = os.path.join(out_dir, "merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("# whisper-boring BPE merges\n")
        for a, b, _rank in merges:
            f.write(f"{_tok_text(a)} {_tok_text(b)}\n")
    print(f"Wrote {len(merges)} merges to {merges_path}")

    print("Special token ids (cross-check against tokenizer.br's TOK_* constants):")
    print(f"  eot={tok.eot}  sot={tok.sot}  en={tok.to_language_token('en')}  "
          f"transcribe={tok.transcribe}  no_timestamps={tok.no_timestamps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Whisper weights to WBRF format")
    parser.add_argument("--model", default="base", choices=list(N_AUDIO_LAYERS),
                        help="Whisper model size")
    parser.add_argument("--out", default="weights/base.wbrf",
                        help="Output WBRF file path")
    parser.add_argument("--tokenizer-out", default="weights/tokenizer",
                        help="Output directory for tokenizer files")
    args = parser.parse_args()

    export(args.model, args.out)
    export_tokenizer(args.model, args.tokenizer_out)
