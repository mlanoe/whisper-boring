# whisper-boring

A port of [Whisper](https://github.com/openai/whisper) — OpenAI's automatic speech recognition model — written in [Boring](https://github.com/mlanoe/boring), a language that compiles to Rust.

The goal is a clean, readable implementation that produces fast, native Rust code and can target multiple compute backends: CPU, CUDA, Metal, and wgpu.

---

## Status

Work in progress. The project structure is in place; the model implementation is the next step.

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
# Interpret directly
boring run examples/transcribe.br

# Transpile to Rust
boring build examples/transcribe.br
cd examples/transcribe_rust && cargo run
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
