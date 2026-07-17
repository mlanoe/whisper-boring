# Changelog

All notable changes to whisper-boring are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Fixed

- **Mel-spectrogram bit-compatibility with Whisper** (`src/audio.br`, `src/audio_gpu.br`, `src/math.br`, `wgpu_test/`) — four formula mismatches identified by `wgpu_test/compare_whisper.py` are corrected, verified numerically against `whisper.audio.mel_filters`/`log_mel_spectrogram` (max abs diff < 2e-9 on the filterbank, < 1.3e-5 on the full pipeline on a real clip):
  - `N_FFT` 512 → 400, with `rfft_windowed` generalised to a direct O(n_fft²) DFT (twiddle-table lookup) so it works for non-power-of-2 sizes, replacing the radix-2-only Cooley-Tukey FFT.
  - `hann_window` switched from symmetric (denominator `n-1`) to periodic (denominator `n`), matching `torch.hann_window`'s default.
  - `mel_filterbank` rewritten to match librosa's Slaney-normalized mel filterbank (continuous Hz ramps + area normalization) instead of an HTK-scale, integer-bin approximation.
  - STFT now drops the last centered frame (`torch.stft(...)[..., :-1]`), and log compression uses `log10` with a fixed `1e-10` floor instead of `10*log10` with a signal-relative floor.
  - The wgpu kernels (`src/audio_gpu.br`, `wgpu_test/shaders/kernels.wgsl`) and both GPU test harnesses (`test/test_mel_gpu.br`, `test/test_gpu_bench.br`) were updated to match.

- **Real weight loading, end-to-end towards a transcription** (`tools/export_weights.py`, `src/tokenizer.br`, `src/main.br`, `src/encoder.br`, `src/decoder.br`, `src/model.br`):
  - `export_weights.py`'s tensor name mapping targeted HuggingFace Transformers' `WhisperModel` state-dict keys (`model.encoder.layers.N.self_attn.q_proj.weight`, ...), but `whisper.load_model()` (the actual `openai-whisper` package) uses its own, different naming (`encoder.blocks.N.attn.query.weight`, ...) — every tensor lookup silently failed. Rewrote the mapping against the real key names; verified all 167/167 tensors resolve for `tiny` (151 MB exported).
  - `export_tokenizer()` called APIs (`.tokenizer.get_vocab()`, `.bpe_ranks`) that no longer exist — current `openai-whisper` uses a `tiktoken` encoding internally. Rewrote it to read `tiktoken`'s byte-sequence→rank table and reconstruct a classic `vocab.json` + `merges.txt` (recovering merge order via the standard tiktoken→BPE-merges algorithm). Round-trip verified against `tokenizer.br`'s actual `decode()` logic.
  - `tokenizer.br`'s hardcoded special-token IDs (`TOK_EOT`, `TOK_SOT`, `TOK_LANG_EN`, `TOK_NO_TIMESTAMPS`) were off by 1-4 against the current multilingual tokenizer's real numbering; corrected and cross-checked against the exporter's own printed values.
  - `src/encoder.br` / `src/decoder.br` had multiple struct fields declared on one line (invalid syntax — one field per line) and a multi-line `guard ... else ...` (also invalid — `else` must be on the same line); both are boring parse errors that blocked `use encoder`/`use decoder` entirely. Fixed.
  - `ModelConfig.for_size()` was declared as an instance method (`req`) but called as a static factory (`ModelConfig.for_size(size)`); needs the `type` qualifier. Fixed.
  - `src/main.br` now parses real CLI args via boring's `args()` (previously a TODO/stub with hardcoded paths) and defaults to the generated `weights/tiny.wbrf`.
  - Added `test/test_wbrf_parse.br` + `test/tiny_test.wbrf`: a small, fast fixture validating the WBRF parsing logic itself, independent of scale.

### Known issues

- The boring interpreter crashes or hangs on arrays larger than roughly 10,000-16,000 elements in some contexts (reproduces on unmodified code; not specific to this project) — tracked separately, outside whisper-boring. Loading a real model's weights via `boring run` now hits this concretely: it fails with a ~52 GB allocation request while parsing the very first multi-thousand-element tensor (`encoder.conv1.weight`, 92,160 floats), so no end-to-end transcription via the interpreter is possible until that lands.
- Independent of that bug, tree-walking a full Whisper forward pass (1500 encoder positions × several transformer layers) in `boring run` will be very slow even once arrays scale correctly — a practical transcription for performance comparison purposes will want the transpiled `--target wgpu` path, not raw interpretation.

---

## [0.2.0] — 2026-07-17

### Added

- **`src/math.br`** — complete linear algebra library: `Complex`, Cooley-Tukey FFT, `softmax`, `layer_norm`, `layer_norm_seq`, `gelu`, `matmul`, `matmul_bt`, `linear`, `conv1d`, `attention`, `mha`.
- **`src/audio.br`** — mel-spectrogram pipeline: Hann window, mel filterbank (hz↔mel conversion, triangular filters), STFT power spectrum, log-mel compression and normalisation, `load_wav` stub.
- **`src/tokenizer.br`** — BPE tokenizer: `encode` (byte-level BPE with merge rules), `decode`, `prompt_tokens`, `load_tokenizer` stub.
- **`src/encoder.br`** — full `AudioEncoder` forward pass: convolutional stem (two Conv1d + GELU, stride-2 downsampling), positional embedding addition, N pre-norm transformer blocks, final layer norm.
- **`src/decoder.br`** — full `TextDecoder` forward pass: token + positional embeddings, N pre-norm decoder blocks (causal self-attention, cross-attention, MLP), weight-tied logit projection, greedy generation loop.
- **`src/model.br`** — `Whisper` struct, `ModelConfig` with `Tiny/Base/Small/Medium` variants, `transcribe` pipeline, `load_whisper` stub.
- **`src/main.br`** — CLI entry point.
- **`examples/transcribe.br`** — smoke-test with a synthetic 440 Hz sine wave (no weights required).
- **`docs/weight-format.md`** — WBRF binary format spec and PyTorch tensor name mapping.

### Pending

- File I/O: `load_wav`, `load_tokenizer`, weight parser (blocked on boring file API).
- `tools/export_weights.py` Python export script.
- Beam search decoding and timestamp prediction.

---

## [0.1.0] — 2026-07-17

### Added

- Initial project structure: `src/`, `examples/`, `docs/`, `LICENSE`, `CLA.md`, CI workflow.
