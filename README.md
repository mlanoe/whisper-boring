# whisper-boring

A port of [Whisper](https://github.com/openai/whisper) — OpenAI's automatic speech recognition model — written in [Boring](https://github.com/mlanoe/boring), a language that compiles to Rust.

The goal is a clean, readable implementation that produces fast, native Rust code and can target multiple compute backends: CPU, CUDA, Metal, and wgpu.

---

## Status

The full pipeline — WAV loading, log-mel spectrogram, encoder, decoder with
KV-cache, greedy decoding, tokenizer — is implemented and runs end-to-end.
`boring build --target metal` (Apple Silicon/macOS) compiles cleanly and
produces correct transcriptions against real OpenAI Whisper weights (`base`
model verified against a real speech sample). CUDA, ROCm, and wgpu targets
share the same `src/*.br` source (GPU kernels are written once, per the
Metal backend doc's `'unified`/`'sync`/`'gpu` qualifiers) but are unverified
on this project specifically — Metal is the actively-tested backend so far,
matching the project's own "compare against reference Whisper" benchmark
goal (`CLAUDE.md`).

The plain `std` target (`boring build`, no `--target`) does **not** build —
this project's GPU-callable kernels have no CPU-only host codegen path in
`boring` today (a real compiler gap, filed upstream). Use `--target metal`
(or another GPU target) instead.

Benchmarking against Python/PyTorch reference Whisper (this project's
stated end goal) hasn't started yet.

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

- [Rust toolchain](https://rustup.rs/) (edition 2021)
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
cd src/main_metal && cargo build
./target/debug/main path/to/audio.wav --model base \
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
│   ├── model.br        # Transformer model definition
│   ├── encoder.br      # Audio encoder
│   ├── decoder.br      # Text decoder
│   ├── tokenizer.br    # Token encoding / decoding
│   ├── audio.br        # Audio loading and mel-spectrogram
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
