# Weight file format — WBRF

whisper-boring uses a simple flat binary format to store model weights.
The file extension is `.wbrf` (Whisper Boring Raw Floats).

## Binary layout

```
magic      : 4 bytes   — 0x57 0x42 0x52 0x46 ("WBRF")
n_tensors  : 4 bytes   — number of tensors (little-endian u32)

for each tensor:
  name_len : 4 bytes   — byte length of the name (LE u32)
  name     : name_len bytes — UTF-8 tensor name
  n_elems  : 4 bytes   — number of float32 elements (LE u32)
  data     : n_elems × 4 bytes — float32 values, little-endian, row-major
```

## Tensor name mapping

| WBRF name | Whisper PyTorch path |
|---|---|
| `encoder.conv1.weight` | `model.encoder.conv1.weight` |
| `encoder.conv1.bias` | `model.encoder.conv1.bias` |
| `encoder.conv2.weight` | `model.encoder.conv2.weight` |
| `encoder.conv2.bias` | `model.encoder.conv2.bias` |
| `encoder.pos_emb` | `model.encoder.embed_positions.weight` |
| `encoder.blocks.N.ln1.weight` | `model.encoder.layers.N.self_attn_layer_norm.weight` |
| `encoder.blocks.N.ln1.bias` | `model.encoder.layers.N.self_attn_layer_norm.bias` |
| `encoder.blocks.N.attn_q.weight` | `model.encoder.layers.N.self_attn.q_proj.weight` |
| `encoder.blocks.N.attn_q.bias` | `model.encoder.layers.N.self_attn.q_proj.bias` |
| `encoder.blocks.N.attn_k.weight` | `model.encoder.layers.N.self_attn.k_proj.weight` |
| `encoder.blocks.N.attn_v.weight` | `model.encoder.layers.N.self_attn.v_proj.weight` |
| `encoder.blocks.N.attn_v.bias` | `model.encoder.layers.N.self_attn.v_proj.bias` |
| `encoder.blocks.N.attn_o.weight` | `model.encoder.layers.N.self_attn.out_proj.weight` |
| `encoder.blocks.N.attn_o.bias` | `model.encoder.layers.N.self_attn.out_proj.bias` |
| `encoder.blocks.N.ln2.weight` | `model.encoder.layers.N.final_layer_norm.weight` |
| `encoder.blocks.N.ln2.bias` | `model.encoder.layers.N.final_layer_norm.bias` |
| `encoder.blocks.N.mlp_fc.weight` | `model.encoder.layers.N.fc1.weight` |
| `encoder.blocks.N.mlp_fc.bias` | `model.encoder.layers.N.fc1.bias` |
| `encoder.blocks.N.mlp_pr.weight` | `model.encoder.layers.N.fc2.weight` |
| `encoder.blocks.N.mlp_pr.bias` | `model.encoder.layers.N.fc2.bias` |
| `encoder.ln.weight` | `model.encoder.layer_norm.weight` |
| `encoder.ln.bias` | `model.encoder.layer_norm.bias` |
| `decoder.tok_emb` | `model.decoder.embed_tokens.weight` |
| `decoder.pos_emb` | `model.decoder.embed_positions.weight` |
| `decoder.blocks.N.ln1.weight` | `model.decoder.layers.N.self_attn_layer_norm.weight` |
| `decoder.blocks.N.sa_q.weight` | `model.decoder.layers.N.self_attn.q_proj.weight` |
| `decoder.blocks.N.sa_k.weight` | `model.decoder.layers.N.self_attn.k_proj.weight` |
| `decoder.blocks.N.sa_v.weight` | `model.decoder.layers.N.self_attn.v_proj.weight` |
| `decoder.blocks.N.sa_o.weight` | `model.decoder.layers.N.self_attn.out_proj.weight` |
| `decoder.blocks.N.ln2.weight` | `model.decoder.layers.N.encoder_attn_layer_norm.weight` |
| `decoder.blocks.N.ca_q.weight` | `model.decoder.layers.N.encoder_attn.q_proj.weight` |
| `decoder.blocks.N.ca_k.weight` | `model.decoder.layers.N.encoder_attn.k_proj.weight` |
| `decoder.blocks.N.ca_v.weight` | `model.decoder.layers.N.encoder_attn.v_proj.weight` |
| `decoder.blocks.N.ca_o.weight` | `model.decoder.layers.N.encoder_attn.out_proj.weight` |
| `decoder.blocks.N.ln3.weight` | `model.decoder.layers.N.final_layer_norm.weight` |
| `decoder.blocks.N.mlp_fc.weight` | `model.decoder.layers.N.fc1.weight` |
| `decoder.blocks.N.mlp_pr.weight` | `model.decoder.layers.N.fc2.weight` |
| `decoder.ln.weight` | `model.decoder.layer_norm.weight` |
| `decoder.ln.bias` | `model.decoder.layer_norm.bias` |

Replace `N` with the layer index (0-based).

## Export script

```python
# tools/export_weights.py
import sys, struct, torch, whisper

model_name = sys.argv[sys.argv.index("--model") + 1]
out_path   = sys.argv[sys.argv.index("--out")   + 1]

model = whisper.load_model(model_name)
sd    = model.state_dict()

# (name mapping omitted for brevity — see full script in tools/)
with open(out_path, "wb") as f:
    f.write(b"WBRF")
    f.write(struct.pack("<I", len(sd)))
    for name, tensor in sd.items():
        data = tensor.float().cpu().numpy().tobytes()
        f.write(struct.pack("<I", len(name.encode())))
        f.write(name.encode())
        f.write(struct.pack("<I", tensor.numel()))
        f.write(data)
```
