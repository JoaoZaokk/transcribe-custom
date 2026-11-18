#!/usr/bin/env python3
"""GGUF v3 surgical modifier for Parakeet Ultra (vocab pruning, logit masking, layer pruning)."""
import argparse
import os
import struct
import sys

GGUF_FMT = {
    0: ("B", 1),
    1: ("b", 1),
    2: ("H", 2),
    3: ("h", 2),
    4: ("I", 4),
    5: ("i", 4),
    6: ("f", 4),
    7: ("?", 1),
    10: ("Q", 8),
    11: ("q", 8),
    12: ("d", 8),
}

# (blck_size, type_size)
GGML_TRAITS = {
    0: (1, 4),     # F32
    1: (1, 2),     # F16
    2: (32, 18),   # Q4_0
    3: (32, 20),   # Q4_1
    6: (32, 22),   # Q5_0
    7: (32, 24),   # Q5_1
    8: (32, 34),   # Q8_0
    10: (256, 84), # Q2_K
    11: (256, 110),# Q3_K
    12: (256, 144),# Q4_K
    13: (256, 176),# Q5_K
    14: (256, 210),# Q6_K
    20: (32, 18),  # IQ4_NL
    23: (256, 136),# IQ4_XS
    30: (1, 2),    # BF16
    39: (32, 17),  # MXFP4 (1 byte e8m0 scale + 16 bytes packed 4-bit = 17 bytes per 32)
}

PT_EN_CHARS = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,!?;:'\"()-/%$@&#+*=_[]{}~`\\|<> \t\n"
    "\u2581"  # SentencePiece word boundary '▁'
    "áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ"
    "–—‘’“”€£°ºª…·•"
)

# Optional common loanwords/foreign chars still seen occasionally in PT/EN technical text: keep pure ASCII + PT-BR accents by default.
def is_pt_en_piece(tok: str) -> bool:
    return all(c in PT_EN_CHARS for c in tok)

def read_str(f) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8")

def write_str(f, s: str):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)

def read_val(f, vt: int):
    if vt == 8:
        return read_str(f)
    if vt == 9:
        et, cnt = struct.unpack("<IQ", f.read(12))
        return (et, [read_val(f, et) for _ in range(cnt)])
    fmt, sz = GGUF_FMT[vt]
    return struct.unpack("<" + fmt, f.read(sz))[0]

def write_val(f, vt: int, val):
    if vt == 8:
        write_str(f, val)
    elif vt == 9:
        et, arr = val
        f.write(struct.pack("<IQ", et, len(arr)))
        for x in arr:
            write_val(f, et, x)
    else:
        fmt, _ = GGUF_FMT[vt]
        f.write(struct.pack("<" + fmt, val))

def tensor_nbytes(dims, gtype: int) -> int:
    blck, tsz = GGML_TRAITS[gtype]
    assert dims[0] % blck == 0, f"dim[0]={dims[0]} not divisible by blck={blck} for type {gtype}"
    row_bytes = (dims[0] // blck) * tsz
    nrows = 1
    for d in dims[1:]:
        nrows *= d
    return row_bytes * nrows

def row_bytes(dim0: int, gtype: int) -> int:
    blck, tsz = GGML_TRAITS[gtype]
    assert dim0 % blck == 0
    return (dim0 // blck) * tsz

def align_up(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--vocab-mode", choices=["none", "mask", "prune"], default="none")
    ap.add_argument("--drop-last-enc-layers", type=int, default=0)
    args = ap.parse_args()

    with open(args.input, "rb") as fin:
        magic = fin.read(4)
        assert magic == b"GGUF"
        ver, n_tensors, n_kv = struct.unpack("<IQQ", fin.read(20))
        assert ver == 3
        kvs = []
        kv_map = {}
        for _ in range(n_kv):
            k = read_str(fin)
            vt = struct.unpack("<I", fin.read(4))[0]
            v = read_val(fin, vt)
            kvs.append([k, vt, v])
            kv_map[k] = len(kvs) - 1

        alignment = 32
        if "general.alignment" in kv_map:
            alignment = kvs[kv_map["general.alignment"]][2]

        tensors = []
        for _ in range(n_tensors):
            name = read_str(fin)
            nd = struct.unpack("<I", fin.read(4))[0]
            dims = list(struct.unpack("<" + "Q" * nd, fin.read(8 * nd)))
            tt, off = struct.unpack("<IQ", fin.read(12))
            tensors.append({"name": name, "dims": dims, "type": tt, "off": off, "nbytes": tensor_nbytes(dims, tt)})

        data_start = align_up(fin.tell(), alignment)

        # Read tensor payloads into memory (models are <= 1.3 GB, fits easily in 16 GB RAM)
        for t in tensors:
            fin.seek(data_start + t["off"])
            t["data"] = bytearray(fin.read(t["nbytes"]))
            assert len(t["data"]) == t["nbytes"]

    # Always set languages metadata to pt, en
    if "general.languages" in kv_map:
        kvs[kv_map["general.languages"]][2] = (8, ["pt", "en"])

    if args.vocab_mode != "none":
        toks = kvs[kv_map["tokenizer.ggml.tokens"]][2][1]
        scores = kvs[kv_map["tokenizer.ggml.scores"]][2][1]
        ttypes = kvs[kv_map["tokenizer.ggml.token_type"]][2][1]
        blank_id = kvs[kv_map["tokenizer.ggml.blank_token_id"]][2]
        assert blank_id == len(toks) - 1

        # Special tokens 0..273 are kept intact; pieces 274..blank_id-1 filtered by PT/EN charset; blank kept at end
        keep_ids = list(range(274))
        drop_ids = []
        for i in range(274, blank_id):
            if is_pt_en_piece(toks[i]):
                keep_ids.append(i)
            else:
                drop_ids.append(i)
        keep_ids.append(blank_id)
        print(f"Vocab analysis: total={len(toks)}, keep={len(keep_ids)}, drop_non_pten={len(drop_ids)}")
    else:
        keep_ids = []
        drop_ids = []

    if args.vocab_mode == "mask":
        # Set joint.out.bias for dropped token indices to -1000.0 (suppress non-PT/EN outputs losslessly)
        for t in tensors:
            if t["name"] == "joint.out.bias":
                assert t["type"] == 0  # F32
                vals = list(struct.unpack("<" + "f" * t["dims"][0], t["data"]))
                for idx in drop_ids:
                    vals[idx] = -1000.0
                t["data"] = bytearray(struct.pack("<" + "f" * len(vals), *vals))
                print(f"Masked {len(drop_ids)} logits in joint.out.bias to -1000.0")
    elif args.vocab_mode == "prune":
        new_toks = [toks[i] for i in keep_ids]
        new_scores = [scores[i] for i in keep_ids]
        new_ttypes = [ttypes[i] for i in keep_ids]
        new_vocab = len(keep_ids)
        new_blank = new_vocab - 1

        kvs[kv_map["tokenizer.ggml.tokens"]][2] = (8, new_toks)
        kvs[kv_map["tokenizer.ggml.scores"]][2] = (6, new_scores)
        kvs[kv_map["tokenizer.ggml.token_type"]][2] = (5, new_ttypes)
        kvs[kv_map["tokenizer.ggml.blank_token_id"]][2] = new_blank
        kvs[kv_map["stt.parakeet.predictor.vocab"]][2] = new_vocab

        # joint has 5 TDT duration outputs at the tail: indices blank_id+1 .. blank_id+5
        n_dur = kvs[kv_map["stt.parakeet.joint.num_extra_outputs"]][2]
        joint_keep = keep_ids + list(range(blank_id + 1, blank_id + 1 + n_dur))

        for t in tensors:
            if t["name"] == "pred.embed.weight":
                rb = row_bytes(t["dims"][0], t["type"])
                out_d = bytearray()
                for idx in keep_ids:
                    out_d.extend(t["data"][idx * rb : (idx + 1) * rb])
                t["data"] = out_d
                t["dims"][1] = new_vocab
                t["nbytes"] = len(out_d)
                print(f"Pruned pred.embed.weight -> {t['dims']} ({t['nbytes']} bytes)")
            elif t["name"] == "joint.out.weight":
                rb = row_bytes(t["dims"][0], t["type"])
                out_d = bytearray()
                for idx in joint_keep:
                    out_d.extend(t["data"][idx * rb : (idx + 1) * rb])
                t["data"] = out_d
                t["dims"][1] = new_vocab + n_dur
                t["nbytes"] = len(out_d)
                print(f"Pruned joint.out.weight -> {t['dims']} ({t['nbytes']} bytes)")
            elif t["name"] == "joint.out.bias":
                assert t["type"] == 0
                vals = struct.unpack("<" + "f" * t["dims"][0], t["data"])
                new_vals = [vals[idx] for idx in joint_keep]
                t["data"] = bytearray(struct.pack("<" + "f" * len(new_vals), *new_vals))
                t["dims"][0] = new_vocab + n_dur
                t["nbytes"] = len(t["data"])
                print(f"Pruned joint.out.bias -> {t['dims']} ({t['nbytes']} bytes)")

    if args.drop_last_enc_layers > 0:
        orig_layers = kvs[kv_map["stt.parakeet.encoder.n_layers"]][2]
        new_layers = orig_layers - args.drop_last_enc_layers
        assert new_layers > 0
        kvs[kv_map["stt.parakeet.encoder.n_layers"]][2] = new_layers
        kept = []
        dropped = 0
        for t in tensors:
            if t["name"].startswith("enc.blocks."):
                blk_idx = int(t["name"].split(".")[2])
                if blk_idx >= new_layers:
                    dropped += 1
                    continue
            kept.append(t)
        tensors = kept
        print(f"Pruned encoder layers: {orig_layers} -> {new_layers} (dropped {dropped} tensors)")

    # Write output GGUF v3
    cur_off = 0
    for t in tensors:
        t["new_off"] = cur_off
        cur_off = align_up(cur_off + t["nbytes"], alignment)

    with open(args.output, "wb") as fout:
        fout.write(b"GGUF")
        fout.write(struct.pack("<IQQ", 3, len(tensors), len(kvs)))
        for k, vt, v in kvs:
            write_str(fout, k)
            fout.write(struct.pack("<I", vt))
            write_val(fout, vt, v)
        for t in tensors:
            write_str(fout, t["name"])
            fout.write(struct.pack("<I", len(t["dims"])))
            fout.write(struct.pack("<" + "Q" * len(t["dims"]), *t["dims"]))
            fout.write(struct.pack("<IQ", t["type"], t["new_off"]))
        pad_hdr = align_up(fout.tell(), alignment) - fout.tell()
        if pad_hdr:
            fout.write(b"\x00" * pad_hdr)
        base_pos = fout.tell()
        for t in tensors:
            pos = fout.tell() - base_pos
            if pos < t["new_off"]:
                fout.write(b"\x00" * (t["new_off"] - pos))
            fout.write(t["data"])
        end_pad = align_up(fout.tell() - base_pos, alignment) - (fout.tell() - base_pos)
        if end_pad:
            fout.write(b"\x00" * end_pad)
    print(f"Wrote {args.output} ({os.path.getsize(args.output)/1024/1024:.2f} MiB)")

if __name__ == "__main__":
    main()
