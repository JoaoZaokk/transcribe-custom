# Apple Silicon (M2) Parakeet Optimizations & Mixed-Precision Tooling

Forked from [`NairoDorian/transcribe.cpp`](https://github.com/NairoDorian/transcribe.cpp) (which tracks upstream [`handy-computer/transcribe.cpp`](https://github.com/handy-computer/transcribe.cpp)).

## Summary of Changes

1. **Unlocked Parakeet Metal GPU TDT/RNN-T Decoder on Apple Silicon (`src/arch/parakeet/model.cpp`, `src/arch/parakeet/decoder.cpp`)**
   - Upstream/fork commit `e5d0af8d` restricted GPU decoding away from `IGPU` / non-discrete devices (`session_device_decoder`), forcing Apple M2 integrated Metal GPUs back onto CPU threads (`resolve_decode_threads(0)` spawning 8 threads and suffering threadpool sync overhead of `5.6s–10.4s` per clip).
   - Included `BackendKind::Metal` in the default GPU decoder path (disable via `TRANSCRIBE_PARAKEET_DEVICE_DECODER=0`), defaulted GPU decoder weights to `native` (`0.0 MiB` extra F32 duplication; override via `TRANSCRIBE_PARAKEET_DEVICE_DECODER_WEIGHTS=f32`), clamped CPU fallback auto-threads to `1` on `__APPLE__`, and set `resolve_joint_batch_w` default to `16`.

2. **Pre-M5 Metal Conformer Attention Default & Precision/KV Overrides (`src/arch/parakeet/encoder.cpp`, `src/conformer/conformer.cpp`)**
   - On `Metal` (`Apple M2`, `MTLGPUFamilyApple8`), Metal FlashAttention is slower than manual F16/F32 attention for Conformer `rel_pos` shapes; defaulted `enc_use_flash = !is_metal_enc` (still overridable via `TRANSCRIBE_FORCE_FLASH=1` or `TRANSCRIBE_ENCODER_FLASH=1`).
   - Added runtime env knobs for empirical testing: `TRANSCRIBE_PREC_F16` and `TRANSCRIBE_KV_TYPE` (`f16|f32|bf16|q8_0|q4_0|q4_1|q5_0|q5_1`).

3. **Unlocked 2D Quantized Pointwise Convolutions Across All Dtypes (`src/arch/parakeet/weights.cpp`, `src/conformer/conformer.cpp`, `src/transcribe-weights-util.h`)**
   - Expanded `TRANSCRIBE_QUANT_LINEAR_TYPES` to include `Q2_K`, `Q3_K`, `IQ4_NL`, `IQ4_XS`, `MXFP4`, and `TQ2_0`.
   - Fixed `GET_PW` and `conv_module` checks (`ggml_is_quantized || ggml_n_dims == 2`) so 2D `[in, out]` pointwise conv weights execute directly through quantized `ggml_mul_mat` kernels for any dtype instead of falling back to F16/F32 copies (`~144 MiB` saved in `parakeet-ultra`).

4. **Mixed Per-Layer & Per-Role Quantizer + GGUF Vocab Surgery (`tools/transcribe-quantize/main.cpp`, `scripts/gguf_surgery.py`)**
   - Extended `transcribe-quantize` with `--squeeze-pw` (squeezes `[1, in, out]` pointwise convs to `[in, out]`) and `--rules <file>` supporting `blocks:<A>-<B>:<role>=<DTYPE>`, `role:<role>=<DTYPE>`, `exact:<tensor>=<DTYPE>`, and substring rules.
   - Added `scripts/gguf_surgery.py` for SentencePiece token filtering (`8193 -> 5519` PT+EN tokens keeping all 274 special tokens and `blank` ID + TDT duration outputs aligned) and optional logit masking.
   - Ready-to-use recipes live under `scripts/quant_recipes/`.

## Verified Measurements on Apple M2 (Mac14,2, 16 GB Unified Memory, Zero Env Overrides)

All models below include PT+EN vocabulary pruning (`8193 -> 5519` tokens; all 24 encoder layers kept intact). Word/char error rates measured on 6 WAVs (PT tech, PT long, PT 45.9s, EN long, EN product names 56.1s, JFK 11s) against pruned `F16`:

| Variant | Size (MiB) | 24.5s PT (`min`) | Enc / Dec (ms) | 45.9s PT (`min`) | 56.1s EN (`min`) | WER vs F16 | CER vs F16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `GOLD_F16_pten` | 1191.91 | 578.8 ms (42.3x) | 427.8 / 124.2 | 1688.9 ms | 2259.4 ms | 0.00% | 0.00% |
| `uni_Q8_0_pw2d` | 634.77 | 632.6 ms (38.7x) | 475.8 / 131.6 | 3936.9 ms | 2475.7 ms | 0.00% | 0.11% |
| `hyb_attnQ8_ff1Q5K_ff2Q4K_pwQ4K` | **430.77** | 697.4 ms (35.1x) | 530.5 / 139.2 | 2267.0 ms | 2628.7 ms | **1.17%** | **0.40%** |
| `mix_sweet_edgeQ5K_415` (per-layer) | 416.77 | **662.2 ms (37.0x)** | 503.9 / 131.1 | 2184.9 ms | **2467.3 ms** | 1.64% | 0.62% |
| `mix_attnQ8_ffnQ4K_pwQ4K` | 406.77 | 707.3 ms (34.6x) | 542.3 / 136.1 | 2149.5 ms | 2525.0 ms | 1.64% | 0.58% |
| `uni_Q4_K_M_pw2d` | 356.45 | 674.4 ms (36.3x) | 514.5 / 132.4 | **2142.3 ms** | 2571.7 ms | 1.41% | 0.55% |
| `uni_MXFP4_pw2d` | 326.64 | 668.9 ms (36.6x) | 498.7 / 137.6 | 2147.4 ms | 2469.3 ms | 2.11% | 0.62% |

Compare against stock pre-optimization defaults on M2 (`hyb` w/ `t=0`, CPU decoder, Metal flash attention): `6239.6 ms` on 24.5s PT (`11870.5 ms` on 45.9s PT) -> **`697.4 ms`** (`~8.9x` faster end-to-end on 24.5s PT).

## Upstream Credits

- Base ASR model: [`moondream/parakeet-ultra`](https://huggingface.co/moondream/parakeet-ultra) (`73175eb`, `CC-BY-4.0`) — built on NVIDIA `parakeet-tdt-0.6b-v3`
- Upstream GGUF conversion & benchmarks: [`handy-computer/parakeet-ultra-gguf`](https://huggingface.co/handy-computer/parakeet-ultra-gguf)
- Upstream inference engine: [`handy-computer/transcribe.cpp`](https://github.com/handy-computer/transcribe.cpp) & fork [`NairoDorian/transcribe.cpp`](https://github.com/NairoDorian/transcribe.cpp) (MIT)
- Desktop dictation app integration: [`cjpais/Handy`](https://github.com/cjpais/Handy) (MIT)
