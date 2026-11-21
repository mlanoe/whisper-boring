# whisper-boring

A port of [Whisper](https://github.com/openai/whisper) — OpenAI's automatic speech recognition model — written in [Boring](https://github.com/mlanoe/boring), a language that compiles to Rust.

The goal is a clean, readable implementation that produces fast, native Rust code and can target multiple compute backends: CPU SIMD, CUDA, Metal, ROCm, and wgpu.

---

## Status

The full pipeline — WAV loading, log-mel spectrogram, encoder, decoder with
GPU-resident KV-cache, greedy decoding, tokenizer — is implemented in `float32`
and runs end-to-end across backends:
- `boring build --target metal` (Apple Silicon/macOS — `simdgroup_matrix` via `kernel.tensors.linear`, FlashAttention / FlashDecode, and command-buffer batching)
- `boring build --target simd` (portable multi-threaded CPU with AVX-512 / AVX2+FMA / NEON runtime detection)
- `boring build --target wgpu` (cross-platform DirectX 12 / Vulkan / Metal)
- `boring build --target cuda` (NVIDIA GPU with cuBLAS SGEMM via `kernel.tensors.linear`)

---

## What is Boring?

Boring is a high-level language that transpiles to idiomatic Rust. It borrows its syntax from Python and Swift, maps every construct directly onto Rust, and produces auditable output you can read, modify, and ship.

```boring
string greet(string? name):
    let who = name else "stranger"
    "Hello, {who}!"
```

See the [Boring repository](https://github.com/mlanoe/boring) for the full language reference.

---

## Prerequisites

- [Rust toolchain](https://rustup.rs/) (edition 2024)
- [boring](https://github.com/mlanoe/boring) installed

```sh
cargo install --git https://github.com/mlanoe/boring
```

---

## Run

```sh
# Smoke test (mel-spectrogram shape only, no weights needed)
boring run examples/transcribe.br

# Full transcription, transpiled to Rust + Metal (project mode, via boring.toml)
boring build --target metal
cd src/main_metal && cargo build --release
./target/release/main-metal path/to/audio.wav --model base \
    --weights weights/base.wbrf --tokenizer weights/tokenizer

# Portable multi-threaded CPU SIMD build (AVX-512 / AVX2+FMA / NEON)
boring build --target simd
cd src/main_simd && cargo build --release
./target/release/main-simd path/to/audio.wav --model base \
    --weights weights/base.wbrf --tokenizer weights/tokenizer
```

Model weights aren't tracked by git — export them from the real
`openai-whisper` package first:

```sh
python tools/export_weights.py --model base --out weights/base.wbrf
```

---

## Repository layout

```
whisper-boring/
├── src/
│   ├── model.br        # Transformer model definition and WBRF weight loading
│   ├── encoder.br      # GPU-resident audio encoder
│   ├── decoder.br      # GPU-resident text decoder with KV-cache
│   ├── math.br         # CPU linear algebra & DFT primitives (float32)
│   ├── math_gpu.br     # GPU kernels (kernel.tensors.linear, FlashAttention, FlashDecode, Warp LayerNorm)
│   ├── tokenizer.br    # BPE token encoding / decoding
│   ├── audio.br        # Audio loading and mel-spectrogram
│   ├── audio_gpu.br    # GPU STFT and mel-spectrogram kernels
│   ├── io.br           # Binary & UTF-8 I/O helpers
│   └── main.br         # CLI entry point
├── examples/
│   └── transcribe.br   # End-to-end transcription example
├── weights/            # Model weight files (not tracked by git)
├── LICENSE
├── CLA.md
└── CLA-signatories.md
```

---

## Contributing

Contributions are welcome.

Before your first Pull Request can be merged, you must sign the [Contributor License Agreement](CLA.md) by adding your name to [`CLA-signatories.md`](CLA-signatories.md) as part of your PR.

---

## License

Copyright (C) 2026 Mickaël LANOË

This program is free software: you can redistribute it and/or modify it under the terms of the
[GNU General Public License v3.0](LICENSE) as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful, but **without any warranty**.
See the LICENSE file for the full terms.

---

*Mickaël LANOË*
