// transcribe-quantize - re-quantize a transcribe.cpp GGUF.
//
// Reads an input GGUF (typically the source/reference-dtype conversion),
// walks every tensor, and writes a new GGUF where each tensor's dtype is
// chosen by a per-preset table. The dequantize → fp32 → requantize path
// uses ggml's reference quantizers (the same ones llama.cpp's
// llama-quantize uses), so the output is bit-compatible with anything
// that consumes standard GGUF blocks.
//
// Quants live in C++ rather than the Python converters because gguf-py
// has no K-quant / IQ-quant support: conversion stays Python (checkpoint →
// reference-dtype GGUF), every derived quant is produced here against libggml.
//
// The tool is intentionally family-aware in exactly one place:
// classify_tensor(). If you change a family tensor catalog, keep this
// policy and the loader's dtype allowlists in sync.
//
// Output writes use gguf_write_to_file() with only_meta=false: the
// gguf_context's tensor list owns its own ggml_context, and tensor data
// is held inside that context, so the writer streams everything in one
// pass without any external file handling.

#include "ggml-alloc.h"
#include "ggml.h"
#include "gguf.h"
#include "policy.h"

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using transcribe::quantize::Bucket;
using transcribe::quantize::classify_tensor;
using transcribe::quantize::find_preset;
using transcribe::quantize::Preset;
using transcribe::quantize::preset_table;
using transcribe::quantize::resolve_target_type;

namespace {

std::string trim_copy(const std::string & s) {
    size_t b = 0;
    while (b < s.size() && std::isspace((unsigned char) s[b])) {
        ++b;
    }
    size_t e = s.size();
    while (e > b && std::isspace((unsigned char) s[e - 1])) {
        --e;
    }
    return s.substr(b, e - b);
}

bool iequals_str(const std::string & a, const char * b) {
    const size_t n = std::strlen(b);
    if (a.size() != n) {
        return false;
    }
    for (size_t idx = 0; idx < n; ++idx) {
        if (std::tolower((unsigned char) a[idx]) != std::tolower((unsigned char) b[idx])) {
            return false;
        }
    }
    return true;
}

bool parse_ggml_type_name(const std::string & tok, ggml_type & out) {
    struct Entry {
        const char * name;
        ggml_type    type;
    };
    static constexpr Entry kTypes[] = {
        { "F32",      GGML_TYPE_F32      },
        { "F16",      GGML_TYPE_F16      },
        { "BF16",     GGML_TYPE_BF16     },
        { "Q8_0",     GGML_TYPE_Q8_0     },
        { "Q6_K",     GGML_TYPE_Q6_K     },
        { "Q5_K",     GGML_TYPE_Q5_K     },
        { "Q5_0",     GGML_TYPE_Q5_0     },
        { "Q5_1",     GGML_TYPE_Q5_1     },
        { "Q4_K",     GGML_TYPE_Q4_K     },
        { "Q4_0",     GGML_TYPE_Q4_0     },
        { "Q4_1",     GGML_TYPE_Q4_1     },
        { "Q3_K",     GGML_TYPE_Q3_K     },
        { "Q2_K",     GGML_TYPE_Q2_K     },
        { "Q2_0",     GGML_TYPE_Q2_0     },
        { "MXFP4",    GGML_TYPE_MXFP4    },
        { "IQ4_NL",   GGML_TYPE_IQ4_NL   },
        { "IQ4_XS",   GGML_TYPE_IQ4_XS   },
        { "TQ2_0",    GGML_TYPE_TQ2_0    },
        { "TQ1_G128", GGML_TYPE_TQ1_G128 },
    };
    for (const Entry & ent : kTypes) {
        if (iequals_str(tok, ent.name)) {
            out = ent.type;
            return true;
        }
    }
    return false;
}

struct MixRule {
    std::string pattern;
    ggml_type   type = GGML_TYPE_COUNT;
};

bool is_conformer_pw_name(const std::string & name) {
    return name.find("enc.blocks.") != std::string::npos &&
           (name.rfind(".conv.pointwise1.weight") != std::string::npos ||
            name.rfind(".conv.pointwise2.weight") != std::string::npos);
}

bool match_block_role(const std::string & rest, const std::string & role) {
    if (role == "all") {
        return (rest.find("ff1.") == 0 || rest.find("ff2.") == 0 || rest.find("attn.") == 0 ||
                rest == "conv.pointwise1.weight" || rest == "conv.pointwise2.weight") &&
               rest.find("pos_bias_") == std::string::npos;
    }
    if (role == "ffn") {
        return rest == "ff1.linear1.weight" || rest == "ff1.linear2.weight" || rest == "ff2.linear1.weight" ||
               rest == "ff2.linear2.weight";
    }
    if (role == "ff1") {
        return rest == "ff1.linear1.weight" || rest == "ff1.linear2.weight";
    }
    if (role == "ff2") {
        return rest == "ff2.linear1.weight" || rest == "ff2.linear2.weight";
    }
    if (role == "attn") {
        return rest.find("attn.linear_") == 0;
    }
    if (role == "attn_qkv") {
        return rest == "attn.linear_q.weight" || rest == "attn.linear_k.weight" || rest == "attn.linear_v.weight";
    }
    if (role == "attn_out") {
        return rest == "attn.linear_out.weight";
    }
    if (role == "attn_pos") {
        return rest == "attn.linear_pos.weight";
    }
    if (role == "pw") {
        return rest == "conv.pointwise1.weight" || rest == "conv.pointwise2.weight";
    }
    if (role == "pw1") {
        return rest == "conv.pointwise1.weight";
    }
    if (role == "pw2") {
        return rest == "conv.pointwise2.weight";
    }
    if (role == "dw") {
        return rest == "conv.depthwise.weight";
    }
    return false;
}

bool match_mix_rule(const std::string & name, Bucket base_bucket, const std::string & pat) {
    if (pat.rfind("force:", 0) == 0) {
        return name.find(pat.substr(6)) != std::string::npos;
    }
    if (pat.rfind("exact:", 0) == 0) {
        return name == pat.substr(6);
    }
    if (base_bucket == Bucket::Norm) {
        return false;
    }
    if (pat.rfind("blocks:", 0) == 0) {
        // blocks:A-B:ROLE or blocks:A:ROLE
        const std::string spec = pat.substr(7);
        const size_t      colon = spec.find(':');
        if (colon == std::string::npos) {
            return false;
        }
        const std::string range_s = spec.substr(0, colon);
        const std::string role_s  = spec.substr(colon + 1);
        int               lo = -1, hi = -1;
        const size_t      dash = range_s.find('-');
        if (dash == std::string::npos) {
            lo = hi = std::atoi(range_s.c_str());
        } else {
            lo = std::atoi(range_s.substr(0, dash).c_str());
            hi = std::atoi(range_s.substr(dash + 1).c_str());
        }
        constexpr const char kBlkPrefix[] = "enc.blocks.";
        if (name.rfind(kBlkPrefix, 0) != 0) {
            return false;
        }
        const size_t start = std::strlen(kBlkPrefix);
        const size_t dot   = name.find('.', start);
        if (dot == std::string::npos) {
            return false;
        }
        const int blk_idx = std::atoi(name.substr(start, dot - start).c_str());
        if (blk_idx < lo || blk_idx > hi) {
            return false;
        }
        return match_block_role(name.substr(dot + 1), role_s);
    }
    if (pat.rfind("role:", 0) == 0) {
        const std::string role = pat.substr(5);
        if (role == "pre_encode.conv") {
            return name.rfind("enc.pre_encode.conv.", 0) == 0 && name.find(".weight") != std::string::npos;
        }
        if (role == "pre_encode.out") {
            return name == "enc.pre_encode.out.weight";
        }
        if (role == "pred.embed") {
            return name == "pred.embed.weight";
        }
        if (role == "pred.lstm") {
            return name.rfind("pred.lstm.", 0) == 0 &&
                   (name.find(".Wx") != std::string::npos || name.find(".Wh") != std::string::npos);
        }
        if (role == "pred") {
            return name.rfind("pred.", 0) == 0;
        }
        if (role == "joint.out") {
            return name == "joint.out.weight";
        }
        if (role == "joint") {
            return name.rfind("joint.", 0) == 0;
        }
        if (role == "vad") {
            return name.rfind("vad.", 0) == 0 && name.find(".weight") != std::string::npos;
        }
        return false;
    }
    return name.find(pat) != std::string::npos;
}

bool load_rules_file(const char * path, std::vector<MixRule> & rules) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::fprintf(stderr, "transcribe-quantize: cannot open rules file: %s\n", path);
        return false;
    }
    std::string line;
    int         lineno = 0;
    while (std::getline(ifs, line)) {
        ++lineno;
        const size_t hash = line.find('#');
        if (hash != std::string::npos) {
            line = line.substr(0, hash);
        }
        line = trim_copy(line);
        if (line.empty()) {
            continue;
        }
        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
            std::fprintf(stderr, "transcribe-quantize: invalid rule at %s:%d (expected pattern=TYPE)\n", path, lineno);
            return false;
        }
        MixRule r;
        r.pattern            = trim_copy(line.substr(0, eq));
        const std::string ts = trim_copy(line.substr(eq + 1));
        if (!parse_ggml_type_name(ts, r.type)) {
            std::fprintf(stderr, "transcribe-quantize: unknown type \"%s\" at %s:%d\n", ts.c_str(), path, lineno);
            return false;
        }
        rules.push_back(std::move(r));
    }
    return true;
}

// ---------------------------------------------------------------------------
// Per-tensor requant
// ---------------------------------------------------------------------------

// Walk type traits to dequantize an arbitrary input tensor into fp32.
// Mirrors decoder.cpp's read_tensor_to_f32 (including the F32 fast
// path that exists because the type_traits[F32] entry leaves to_float
// null upstream — F32 dequant is the identity).
bool dequantize_to_f32(const ggml_tensor * t, std::vector<float> & out) {
    const int64_t nelem = ggml_nelements(t);
    if (nelem <= 0) {
        std::fprintf(stderr, "transcribe-quantize: tensor \"%s\" has nelem=%lld\n", t->name, (long long) nelem);
        return false;
    }
    out.resize((size_t) nelem);

    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, (size_t) nelem * sizeof(float));
        return true;
    }
    const auto * tt = ggml_get_type_traits(t->type);
    if (tt == nullptr || tt->to_float == nullptr) {
        std::fprintf(stderr, "transcribe-quantize: type %s has no to_float\n", ggml_type_name(t->type));
        return false;
    }
    tt->to_float(t->data, out.data(), nelem);
    return true;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

struct PerTypeStats {
    int64_t n_tensors = 0;
    int64_t bytes     = 0;
};

const char * type_label(ggml_type t) {
    return ggml_type_name(t);
}

void print_stats(const std::map<ggml_type, PerTypeStats> & stats, const char * label) {
    int64_t total_bytes = 0;
    int64_t total_n     = 0;
    std::printf("  %-12s%-10s%s\n", "type", "tensors", "MB");
    for (const auto & kv : stats) {
        std::printf("  %-12s%-10lld%.1f\n", type_label(kv.first), (long long) kv.second.n_tensors,
                    (double) kv.second.bytes / (1024.0 * 1024.0));
        total_bytes += kv.second.bytes;
        total_n += kv.second.n_tensors;
    }
    std::printf("  %-12s%-10lld%.1f total (%s)\n", "", (long long) total_n, (double) total_bytes / (1024.0 * 1024.0),
                label);
}

// ---------------------------------------------------------------------------
// Bulk-KV trailer
// ---------------------------------------------------------------------------

// Large tokenizer KVs, in canonical trailer order. Mirrors BULK_KV_KEYS in
// scripts/lib/gguf_common.py: the Python converter relocates these to the end
// of the KV section so range-read consumers can fetch the small scalar
// metadata (general.*, stt.*, tokenizer identity) without pulling the multi-MB
// tokenizer tables. We re-assert that layout here because the quantizer's
// general.file_type override appends via ggml's remove-then-append setter,
// which would otherwise strand file_type — a tiny scalar a remote consumer
// wants — *after* the tokenizer trailer.
const char * const kBulkKvKeys[] = {
    "tokenizer.ggml.tokens", "tokenizer.ggml.scores",   "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges", "tokenizer.chat_template",
};

// Byte size of one element for the scalar element types that can appear inside
// a bulk array (strings are handled separately). gguf_type_size is not part of
// the public gguf.h, so we size the handful of types we actually move; 0 means
// "unexpected type, leave the key where it is."
size_t bulk_kv_elem_size(gguf_type t) {
    switch (t) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return 8;
        default:
            return 0;
    }
}

// Move the bulk tokenizer KVs to the very end of ctx's KV section, preserving
// each value. Every ggml gguf setter is remove-then-append, so reading a key's
// value into a local copy and re-setting it relocates it to the tail. Keys are
// processed in kBulkKvKeys order, so the trailer ends up canonical regardless
// of how they were interleaved on the way in. Absent keys are skipped. Mirrors
// move_bulk_metadata_last() in scripts/lib/gguf_common.py.
void move_bulk_kv_last(gguf_context * ctx) {
    for (const char * key : kBulkKvKeys) {
        const int64_t kid = gguf_find_key(ctx, key);
        if (kid < 0) {
            continue;  // not present — nothing to move
        }
        const gguf_type kv_type = gguf_get_kv_type(ctx, kid);
        if (kv_type == GGUF_TYPE_ARRAY) {
            const gguf_type elem = gguf_get_arr_type(ctx, kid);
            const size_t    n    = gguf_get_arr_n(ctx, kid);
            if (elem == GGUF_TYPE_STRING) {
                // Copy the strings out before the remove-then-append re-set
                // frees the originals.
                std::vector<std::string>  store(n);
                std::vector<const char *> ptrs(n);
                for (size_t i = 0; i < n; ++i) {
                    store[i] = gguf_get_arr_str(ctx, kid, i);
                }
                for (size_t i = 0; i < n; ++i) {
                    ptrs[i] = store[i].c_str();
                }
                gguf_set_arr_str(ctx, key, ptrs.data(), n);
            } else {
                const size_t esz = bulk_kv_elem_size(elem);
                if (esz == 0) {
                    continue;  // unexpected element type — leave it in place
                }
                // Copy the raw element bytes out before the re-set frees them.
                const void *        data = gguf_get_arr_data(ctx, kid);
                std::vector<int8_t> tmp(n * esz);
                if (!tmp.empty()) {
                    std::memcpy(tmp.data(), data, n * esz);
                }
                gguf_set_arr_data(ctx, key, elem, tmp.data(), n);
            }
        } else if (kv_type == GGUF_TYPE_STRING) {
            const std::string val = gguf_get_val_str(ctx, kid);
            gguf_set_val_str(ctx, key, val.c_str());
        }
        // Other scalar types are not expected among the bulk keys; ignore.
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void print_usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s INPUT.gguf OUTPUT.gguf [--quant PRESET] [--rules FILE] [--squeeze-pw]\n"
                 "\n"
                 "  --quant PRESET   one of:",
                 argv0);
    size_t         n       = 0;
    const Preset * presets = preset_table(n);
    for (size_t i = 0; i < n; ++i) {
        std::fprintf(stderr, " %s", presets[i].name);
    }
    std::fprintf(stderr, "\n  --rules FILE     mixed-precision rules file (pattern=TYPE)\n"
                         "  --squeeze-pw     squeeze [1, in, out] conv pointwise weights to 2D [in, out]\n");
}

}  // namespace

int main(int argc, char ** argv) {
    const char *         in_path    = nullptr;
    const char *         out_path   = nullptr;
    const char *         quant      = nullptr;
    const char *         rules_path = nullptr;
    bool                 squeeze_pw = false;
    std::vector<MixRule> mix_rules;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--quant" || a == "-q") {
            if (i + 1 >= argc) {
                print_usage(argv[0]);
                return 2;
            }
            quant = argv[++i];
        } else if (a == "--rules" || a == "-r") {
            if (i + 1 >= argc) {
                print_usage(argv[0]);
                return 2;
            }
            rules_path = argv[++i];
        } else if (a == "--squeeze-pw") {
            squeeze_pw = true;
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (in_path == nullptr) {
            in_path = argv[i];
        } else if (out_path == nullptr) {
            out_path = argv[i];
        } else {
            std::fprintf(stderr, "transcribe-quantize: unexpected argument %s\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }
    if (in_path == nullptr || out_path == nullptr || (quant == nullptr && rules_path == nullptr)) {
        print_usage(argv[0]);
        return 2;
    }
    if (quant == nullptr) {
        quant = "F16";
    }
    if (rules_path != nullptr && !load_rules_file(rules_path, mix_rules)) {
        return 2;
    }
    const Preset * preset = find_preset(quant);
    if (preset == nullptr) {
        std::fprintf(stderr, "transcribe-quantize: unknown preset %s\n", quant);
        print_usage(argv[0]);
        return 2;
    }

    std::printf("transcribe-quantize: %s -> %s (preset %s%s%s)\n", in_path, out_path, preset->name,
                rules_path ? ", mixed rules" : "", squeeze_pw ? ", squeeze-pw" : "");

    // Load input gguf with tensor data into a ggml_context.
    //
    // no_alloc=false + ctx=&ctx_in lets ggml allocate one big buffer
    // and load every tensor's bytes into it. After this returns, every
    // ggml_tensor in ctx_in has a valid `data` pointer pointing into
    // that buffer. The data lives until ggml_free(ctx_in).
    ggml_context *   ctx_in = nullptr;
    gguf_init_params in_params{};
    in_params.no_alloc     = false;
    in_params.ctx          = &ctx_in;
    gguf_context * gguf_in = gguf_init_from_file(in_path, in_params);
    if (gguf_in == nullptr) {
        std::fprintf(stderr, "transcribe-quantize: failed to read %s\n", in_path);
        return 1;
    }

    const int64_t n_tensors = gguf_get_n_tensors(gguf_in);
    std::printf("input: %lld tensors\n", (long long) n_tensors);

    // Plan: per-tensor target type + total output buffer size.
    //
    // We need a fresh ggml_context for the output tensors that owns
    // their (possibly newly-quantized) data. The size is the sum of
    // ggml_row_size() across the chosen output types, plus per-tensor
    // metadata overhead, plus alignment slack.
    //
    // Iterate via the gguf side rather than ggml's object list:
    // when ggml_init_from_file uses no_alloc=false, gguf.cpp allocates
    // a single i8 "GGUF tensor data binary blob" inside ctx_in to
    // hold the raw bytes (gguf.cpp:789), then makes every real tensor
    // a view into it. ggml_get_first_tensor would return that blob
    // first; using gguf_get_tensor_name + ggml_get_tensor walks only
    // the named gguf tensors and skips the blob.
    struct PlanEntry {
        ggml_tensor * src;
        ggml_type     dst_type;
        size_t        dst_nbytes;  // ggml_row_size(dst_type, ne0) * nrows
        int           dst_ndims;
        int64_t       dst_ne[GGML_MAX_DIMS];
    };

    std::vector<PlanEntry> plan;
    plan.reserve((size_t) n_tensors);
    size_t                            total_data_bytes = 0;
    std::map<ggml_type, PerTypeStats> in_stats, out_stats;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char *  tname = gguf_get_tensor_name(gguf_in, i);
        ggml_tensor * t     = ggml_get_tensor(ctx_in, tname);
        if (t == nullptr) {
            std::fprintf(stderr, "transcribe-quantize: tensor %s missing from ctx_in\n", tname);
            ggml_free(ctx_in);
            gguf_free(gguf_in);
            return 1;
        }
        const std::string name(t->name);
        const Bucket      base_bucket = classify_tensor(name, t->ne[0]);
        const bool        is_pw3d     = is_conformer_pw_name(name) && t->ne[0] == 1 && t->ne[2] > 1;
        const int64_t     plan_ne0    = (squeeze_pw && is_pw3d) ? t->ne[1] : t->ne[0];

        ggml_type dst_type = GGML_TYPE_COUNT;
        if (t->type == GGML_TYPE_TQ1_G128) {
            dst_type = GGML_TYPE_TQ1_G128;
        } else {
            for (const MixRule & r : mix_rules) {
                if (match_mix_rule(name, base_bucket, r.pattern)) {
                    dst_type = r.type;
                    break;
                }
            }
            if (dst_type == GGML_TYPE_COUNT) {
                dst_type = resolve_target_type(*preset, name, plan_ne0);
            }
        }

        const bool want_2d_pw =
            is_pw3d && (squeeze_pw || ggml_is_quantized(dst_type) || dst_type == GGML_TYPE_BF16);

        int     dst_ndims              = ggml_n_dims(t);
        int64_t dst_ne[GGML_MAX_DIMS]  = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
        if (want_2d_pw) {
            dst_ndims = 2;
            dst_ne[0] = t->ne[1];
            dst_ne[1] = t->ne[2];
            dst_ne[2] = 1;
            dst_ne[3] = 1;
        }

        if (const int64_t blk = ggml_blck_size(dst_type); blk > 1 && (dst_ne[0] % blk) != 0) {
            dst_type = (dst_ne[0] % 32 == 0) ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
        }

        int64_t nrows = 1;
        for (int d = 1; d < GGML_MAX_DIMS; ++d) {
            nrows *= dst_ne[d];
        }
        const size_t row_size = ggml_row_size(dst_type, dst_ne[0]);
        const size_t nb       = row_size * (size_t) nrows;

        PlanEntry pe_entry{};
        pe_entry.src        = t;
        pe_entry.dst_type   = dst_type;
        pe_entry.dst_nbytes = nb;
        pe_entry.dst_ndims  = dst_ndims;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            pe_entry.dst_ne[d] = dst_ne[d];
        }
        plan.push_back(pe_entry);
        total_data_bytes += nb;
        // Round up to ggml's alignment so each tensor starts on a
        // 32-byte boundary inside the output context buffer.
        total_data_bytes = (total_data_bytes + 31) & ~size_t{ 31 };

        in_stats[t->type].n_tensors += 1;
        in_stats[t->type].bytes += (int64_t) ggml_nbytes(t);
        out_stats[dst_type].n_tensors += 1;
        out_stats[dst_type].bytes += (int64_t) nb;
    }

    // Per-tensor metadata overhead (ggml_tensor struct + bookkeeping
    // inside the context). ggml_tensor_overhead() is the canonical
    // accessor; multiply by tensor count + slack.
    const size_t per_tensor_overhead = ggml_tensor_overhead();
    const size_t mem_size            = total_data_bytes + (size_t) n_tensors * per_tensor_overhead +
                                       16 * 1024 * 1024;  // 16 MB safety pad for alignment + bookkeeping

    std::printf("plan: %.1f MB tensor data, %.1f MB context buffer\n", (double) total_data_bytes / (1024.0 * 1024.0),
                (double) mem_size / (1024.0 * 1024.0));
    std::printf("\ninput tensor types:\n");
    print_stats(in_stats, "input");
    std::printf("\noutput tensor types:\n");
    print_stats(out_stats, "output");
    std::printf("\n");

    // Allocate the output ggml_context + gguf_context.
    ggml_init_params out_init{};
    out_init.mem_size      = mem_size;
    out_init.mem_buffer    = nullptr;  // ggml will malloc internally
    out_init.no_alloc      = false;
    ggml_context * ctx_out = ggml_init(out_init);
    if (ctx_out == nullptr) {
        std::fprintf(stderr, "transcribe-quantize: ggml_init(out) failed\n");
        gguf_free(gguf_in);
        return 1;
    }

    gguf_context * gguf_out = gguf_init_empty();
    if (gguf_out == nullptr) {
        std::fprintf(stderr, "transcribe-quantize: gguf_init_empty failed\n");
        ggml_free(ctx_out);
        gguf_free(gguf_in);
        return 1;
    }

    // Copy every KV from the input. This pulls general.architecture,
    // tokenizer.*, stt.parakeet.*, stt.frontend.*, stt.capability.*,
    // and the existing general.file_type if present.
    gguf_set_kv(gguf_out, gguf_in);

    // Override general.file_type with the new preset's tag. The C++
    // loader doesn't read this today, but downstream tools like
    // gguf-dump display it.
    gguf_set_val_u32(gguf_out, "general.file_type", preset->file_type);

    // Re-assert the bulk-KV trailer so general.file_type (and every other
    // small scalar) precedes the multi-MB tokenizer tables. The override
    // above appends file_type last via ggml's remove-then-append setter;
    // without this, a range-read consumer couldn't reach file_type without
    // parsing the whole tokenizer blob. Mirrors the converter's gguf_writer()
    // layout (scripts/lib/gguf_common.py).
    move_bulk_kv_last(gguf_out);

    // Per-tensor: dequant → fp32 → requant → add to ctx_out.
    std::vector<float> fp32_scratch;
    int64_t            requantized = 0;
    int64_t            copied      = 0;

    for (const PlanEntry & e : plan) {
        ggml_tensor * src = e.src;

        // Allocate the destination tensor in ctx_out with the new
        // dtype and the resolved (possibly 2D-squeezed) shape.
        ggml_tensor * dst = ggml_new_tensor(ctx_out, e.dst_type, e.dst_ndims, e.dst_ne);
        if (dst == nullptr) {
            std::fprintf(stderr,
                         "transcribe-quantize: ggml_new_tensor failed for %s "
                         "(out of context memory?)\n",
                         src->name);
            ggml_free(ctx_out);
            gguf_free(gguf_in);
            gguf_free(gguf_out);
            return 1;
        }
        ggml_set_name(dst, src->name);

        // Cheap path: same dtype as source, just memcpy the bytes.
        // Squeezing [1, in, out] -> [in, out] preserves raw element order.
        if (src->type == dst->type) {
            std::memcpy(dst->data, src->data, ggml_nbytes(src));
            ++copied;
        } else {
            // Real requant. Step 1: dequantize the source to fp32 in
            // a host scratch buffer.
            if (!dequantize_to_f32(src, fp32_scratch)) {
                ggml_free(ctx_out);
                gguf_free(gguf_in);
                gguf_free(gguf_out);
                return 1;
            }

            // Step 2: quantize fp32 → dst_type into the destination
            // tensor's data buffer. ggml_quantize_chunk takes (type,
            // src_fp32, dst_bytes, start_elem, nrows, n_per_row,
            // imatrix).
            if (ggml_quantize_requires_imatrix(dst->type)) {
                std::fprintf(stderr,
                             "transcribe-quantize: type %s requires imatrix; "
                             "this tool does not support imatrix quants yet\n",
                             ggml_type_name(dst->type));
                ggml_free(ctx_out);
                gguf_free(gguf_in);
                gguf_free(gguf_out);
                return 1;
            }
            const int64_t n_per_row = e.dst_ne[0];
            int64_t       nrows     = 1;
            for (int d = 1; d < GGML_MAX_DIMS; ++d) {
                nrows *= e.dst_ne[d];
            }
            ggml_quantize_chunk(dst->type, fp32_scratch.data(), dst->data,
                                /*start=*/0, nrows, n_per_row,
                                /*imatrix=*/nullptr);
            ++requantized;
        }

        gguf_add_tensor(gguf_out, dst);
    }

    std::printf("processed: %lld requantized, %lld copied\n", (long long) requantized, (long long) copied);

    // Write the output gguf.
    if (!gguf_write_to_file(gguf_out, out_path, /*only_meta=*/false)) {
        std::fprintf(stderr, "transcribe-quantize: gguf_write_to_file failed\n");
        ggml_free(ctx_out);
        gguf_free(gguf_in);
        gguf_free(gguf_out);
        return 1;
    }

    // Final cleanup. Free in reverse order: gguf_out and ctx_out are
    // independent; gguf_in owns ctx_in (allocated via in_params.ctx)
    // and freeing the gguf context releases the ggml context too.
    ggml_free(ctx_out);
    gguf_free(gguf_out);
    gguf_free(gguf_in);
    ggml_quantize_free();

    std::printf("done. wrote %s\n", out_path);
    return 0;
}
