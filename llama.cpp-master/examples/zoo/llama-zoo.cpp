#include "llama.h"
#include "ggml.h"

#include <vector>
#include <string>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <functional>
#include <thread>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <numeric>

#include "ggml-impl.h"

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include <arm_neon.h>
#endif

static const int N_WORKERS = std::max(1, (int)std::thread::hardware_concurrency() - 2);

#define private public
#define protected public
#include "src/llama-context.h"
#include "src/llama-model.h"
#include "src/llama-memory.h"
#include "src/llama-kv-cache.h"
#undef private
#undef protected

static float fp16_to_fp32(ggml_fp16_t val) {
    return ggml_fp16_to_fp32(val);
}

static ggml_fp16_t fp32_to_fp16(float val) {
    return ggml_fp32_to_fp16(val);
}

using Clock = std::chrono::steady_clock;

static inline double ms_since(const Clock::time_point & t0, const Clock::time_point & t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct Xoshiro256pp {
    uint64_t s[4];

    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    static inline uint64_t splitmix64(uint64_t & x) {
        uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    explicit Xoshiro256pp(uint64_t seed) {
        uint64_t x = seed;
        for (int i = 0; i < 4; ++i) {
            s[i] = splitmix64(x);
        }
    }

    inline uint64_t next_u64() {
        const uint64_t result = rotl(s[0] + s[3], 23) + s[0];

        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    inline uint32_t next_u32() {
        return (uint32_t) next_u64();
    }

    inline float next_f32() {
        const uint32_t u = next_u32();
        return (u + 1.0f) * (1.0f / 4294967296.0f);
    }
};

struct ZigguratNormal {
    static constexpr int kN = 128;
    static uint32_t kn[kN];
    static float wn[kN];
    static float fn[kN];
    static std::atomic<bool> inited;

    static void init() {
        bool expected = false;
        if (!inited.compare_exchange_strong(expected, true)) {
            return;
        }

        const float m1 = 2147483648.0f; // 2^31
        float dn = 3.442619855899f;     // R for 128 layers
        float tn = dn;
        const float vn = 9.91256303526217e-3f; // V
        const float q = vn / std::exp(-0.5f * dn * dn);

        kn[0] = (uint32_t)((dn / q) * m1);
        kn[1] = 0;

        wn[0] = q / m1;
        wn[kN - 1] = dn / m1;

        fn[0] = 1.0f;
        fn[kN - 1] = std::exp(-0.5f * dn * dn);

        for (int i = kN - 2; i >= 1; --i) {
            dn = std::sqrt(-2.0f * std::log(vn / dn + std::exp(-0.5f * dn * dn)));
            kn[i + 1] = (uint32_t)((dn / tn) * m1);
            tn = dn;
            fn[i] = std::exp(-0.5f * dn * dn);
            wn[i] = dn / m1;
        }
    }

    template <class RNG>
    static inline float nfix(int32_t hz, uint32_t iz, RNG & rng) {
        const float r = 3.442619855899f;
        for (;;) {
            if (iz == 0) {
                float x, y;
                do {
                    x = -std::log(rng.next_f32()) / r;
                    y = -std::log(rng.next_f32());
                } while (y + y < x * x);
                return (hz > 0) ? (r + x) : (-r - x);
            }
            const float x = hz * wn[iz];
            if (fn[iz] + rng.next_f32() * (fn[iz - 1] - fn[iz]) < std::exp(-0.5f * x * x)) {
                return x;
            }
            hz = (int32_t) rng.next_u32();
            iz = (uint32_t) hz & (kN - 1);
            const int64_t ahz = (hz < 0) ? -(int64_t)hz : (int64_t)hz;
            if ((uint32_t)ahz < kn[iz]) {
                return hz * wn[iz];
            }
        }
    }

    template <class RNG>
    static inline float sample(RNG & rng) {
        const int32_t hz = (int32_t) rng.next_u32();
        const uint32_t iz = (uint32_t) hz & (kN - 1);
        const int64_t ahz = (hz < 0) ? -(int64_t)hz : (int64_t)hz;
        if ((uint32_t)ahz < kn[iz]) {
            return hz * wn[iz];
        }
        return nfix(hz, iz, rng);
    }
};

uint32_t ZigguratNormal::kn[ZigguratNormal::kN];
float ZigguratNormal::wn[ZigguratNormal::kN];
float ZigguratNormal::fn[ZigguratNormal::kN];
std::atomic<bool> ZigguratNormal::inited{false};

struct StepTimes {
    double t_pos_total_ms      = 0.0;
    double t_pos_gen_ms        = 0.0;
    double t_pos_apply_ms      = 0.0;
    double t_pos_rc_gen_ms     = 0.0;
    double t_pos_rc_apply_ms   = 0.0;
    double t_clear_pos_ms      = 0.0;
    double t_decode_pos_ms     = 0.0;

    double t_neg_total_ms      = 0.0;
    double t_neg_gen_ms        = 0.0;
    double t_neg_apply_ms      = 0.0;
    double t_clear_neg_ms      = 0.0;
    double t_decode_neg_ms     = 0.0;

    double t_update_total_ms   = 0.0;
    double t_update_gen_ms     = 0.0;
    double t_update_apply_ms   = 0.0;

    double t_step_total_ms     = 0.0;

    void print(int step) const {
        printf(
            "[TIME] Step %d | total %.3f ms\n"
            "      +perturb: total %.3f ms | gen(max): %.3f ms | apply(max): %.3f ms | rc_gen(max): %.3f ms | rc_apply(max): %.3f ms | +clear: %.3f ms | +decode: %.3f ms\n"
            "      -perturb: total %.3f ms | gen(max): %.3f ms | apply(max): %.3f ms | -clear: %.3f ms | -decode: %.3f ms\n"
            "      update:   total %.3f ms | gen(max): %.3f ms | apply(max): %.3f ms\n",
            step, t_step_total_ms,
            t_pos_total_ms, t_pos_gen_ms, t_pos_apply_ms, t_pos_rc_gen_ms, t_pos_rc_apply_ms, t_clear_pos_ms, t_decode_pos_ms,
            t_neg_total_ms, t_neg_gen_ms, t_neg_apply_ms, t_clear_neg_ms, t_decode_neg_ms,
            t_update_total_ms, t_update_gen_ms, t_update_apply_ms
        );
    }
};

struct LayerInfo {
    std::string name;
    ggml_tensor* tensor;
    int64_t count;
    uint64_t name_hash;
    bool use_rc = false;
    int64_t n_rows = 0;
    int64_t n_cols = 0;
    std::vector<ggml_fp16_t> row_noise;
    std::vector<ggml_fp16_t> col_noise;
    bool has_rc_cache = false;
};

static bool ensure_tensor_f16(ggml_tensor * t) {
    if (t->type == GGML_TYPE_F16) {
        return true;
    }
    if (t->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t n = ggml_nelements(t);
    ggml_fp16_t * dst = (ggml_fp16_t *) ggml_aligned_malloc((size_t) n * sizeof(ggml_fp16_t));
    if (!dst) {
        return false;
    }

    const float * src = (const float *) t->data;
    for (int64_t i = 0; i < n; ++i) {
        dst[i] = ggml_fp32_to_fp16(src[i]);
    }

    t->data = dst;
    t->type = GGML_TYPE_F16;
    t->nb[0] = ggml_type_size(t->type);
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        t->nb[i] = t->nb[i - 1] * t->ne[i - 1];
    }

    return true;
}

void apply_perturbation_in_place_parallel(
    std::vector<LayerInfo>& layers,
    uint64_t step_seed,
    float scale,
    double & gen_ms,
    double & apply_ms,
    bool cache_rc_noise,
    double & rc_gen_ms,
    double & rc_apply_ms
) {
    if (scale == 0.0f) return;

    ZigguratNormal::init();

    std::atomic<size_t> layer_idx(0);
    std::vector<std::thread> workers;
    std::atomic<int64_t> gen_us_max(0);
    std::atomic<int64_t> apply_us_max(0);
    std::atomic<int64_t> rc_gen_us_max(0);
    std::atomic<int64_t> rc_apply_us_max(0);

    for (int t = 0; t < N_WORKERS; ++t) {
        workers.emplace_back([&]() {
            int64_t local_gen_us = 0;
            int64_t local_apply_us = 0;
            int64_t local_rc_gen_us = 0;
            int64_t local_rc_apply_us = 0;
            std::vector<ggml_fp16_t> noise_buffer;
            while (true) {
                size_t i = layer_idx.fetch_add(1);
                if (i >= layers.size()) break;

                auto& layer = layers[i];
                uint64_t layer_seed = step_seed + layer.name_hash;
                Xoshiro256pp rng(layer_seed);

                ggml_fp16_t* data_ptr = (ggml_fp16_t*)layer.tensor->data;
                int64_t count = layer.count;

                if (layer.use_rc) {
                    if (cache_rc_noise || !layer.has_rc_cache) {
                        auto t_rcg0 = Clock::now();
                        if (layer.row_noise.size() < (size_t)layer.n_rows) {
                            layer.row_noise.resize((size_t)layer.n_rows);
                        }
                        if (layer.col_noise.size() < (size_t)layer.n_cols) {
                            layer.col_noise.resize((size_t)layer.n_cols);
                        }
                        for (int64_t r = 0; r < layer.n_rows; ++r) {
                            float n = ZigguratNormal::sample(rng);
                            layer.row_noise[r] = fp32_to_fp16(n);
                        }
                        for (int64_t c = 0; c < layer.n_cols; ++c) {
                            float n = ZigguratNormal::sample(rng);
                            layer.col_noise[c] = fp32_to_fp16(n);
                        }
                        auto t_rcg1 = Clock::now();
                        local_rc_gen_us += (int64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_rcg1 - t_rcg0).count();
                        layer.has_rc_cache = true;
                    }

                    auto t_rca0 = Clock::now();
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                    for (int64_t r = 0; r < layer.n_rows; ++r) {
                        const float row_scale = fp16_to_fp32(layer.row_noise[r]) * scale;
                        const __fp16 row_hs = (const __fp16) row_scale;
                        const float16x8_t row_vec = vdupq_n_f16(row_hs);
                        const __fp16 * col_ptr = (const __fp16 *) layer.col_noise.data();
                        __fp16 * wptr = (__fp16 *) data_ptr + r * layer.n_cols;
                        int64_t c = 0;
                        for (; c + 8 <= layer.n_cols; c += 8) {
                            float16x8_t w = vld1q_f16(wptr + c);
                            float16x8_t col = vld1q_f16(col_ptr + c);
                            float16x8_t n = vmulq_f16(col, row_vec);
                            w = vaddq_f16(w, n);
                            vst1q_f16(wptr + c, w);
                        }
                        for (; c < layer.n_cols; ++c) {
                            float val = fp16_to_fp32(wptr[c]) + row_scale * fp16_to_fp32(layer.col_noise[c]);
                            wptr[c] = fp32_to_fp16(val);
                        }
                    }
#else
                    for (int64_t r = 0; r < layer.n_rows; ++r) {
                        const float row_v = fp16_to_fp32(layer.row_noise[r]) * scale;
                        ggml_fp16_t * wptr = data_ptr + r * layer.n_cols;
                        for (int64_t c = 0; c < layer.n_cols; ++c) {
                            float val = fp16_to_fp32(wptr[c]) + row_v * fp16_to_fp32(layer.col_noise[c]);
                            wptr[c] = fp32_to_fp16(val);
                        }
                    }
#endif
                    auto t_rca1 = Clock::now();
                    const int64_t rc_apply_us = (int64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_rca1 - t_rca0).count();
                    local_rc_apply_us += rc_apply_us;
                    local_apply_us += rc_apply_us;
                } else {
                    if (noise_buffer.size() < (size_t)count) {
                        noise_buffer.resize((size_t)count);
                    }

                    auto t_gen0 = Clock::now();
                    for (int64_t j = 0; j < count; ++j) {
                        float n = ZigguratNormal::sample(rng) * scale;
                        noise_buffer[j] = fp32_to_fp16(n);
                    }
                    auto t_gen1 = Clock::now();
                    local_gen_us += (int64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_gen1 - t_gen0).count();

                    auto t_apply0 = Clock::now();
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                    int64_t j = 0;
                    const __fp16 * nb = (const __fp16 *) noise_buffer.data();
                    __fp16 * wp = (__fp16 *) data_ptr;
                    for (; j + 8 <= count; j += 8) {
                        float16x8_t w = vld1q_f16(wp + j);
                        float16x8_t n = vld1q_f16(nb + j);
                        w = vaddq_f16(w, n);
                        vst1q_f16(wp + j, w);
                    }
                    for (; j < count; ++j) {
                        float val = fp16_to_fp32(data_ptr[j]) + fp16_to_fp32(noise_buffer[j]);
                        data_ptr[j] = fp32_to_fp16(val);
                    }
#else
                    for (int64_t j = 0; j < count; ++j) {
                        float val = fp16_to_fp32(data_ptr[j]) + fp16_to_fp32(noise_buffer[j]);
                        data_ptr[j] = fp32_to_fp16(val);
                    }
#endif
                    auto t_apply1 = Clock::now();
                    local_apply_us += (int64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_apply1 - t_apply0).count();
                }
            }
            int64_t prev = gen_us_max.load();
            while (prev < local_gen_us && !gen_us_max.compare_exchange_weak(prev, local_gen_us)) {}
            prev = apply_us_max.load();
            while (prev < local_apply_us && !apply_us_max.compare_exchange_weak(prev, local_apply_us)) {}
            prev = rc_gen_us_max.load();
            while (prev < local_rc_gen_us && !rc_gen_us_max.compare_exchange_weak(prev, local_rc_gen_us)) {}
            prev = rc_apply_us_max.load();
            while (prev < local_rc_apply_us && !rc_apply_us_max.compare_exchange_weak(prev, local_rc_apply_us)) {}
        });
    }

    for (auto& w : workers) w.join();

    const int64_t gen_us = gen_us_max.load();
    const int64_t rc_gen_us = rc_gen_us_max.load();
    gen_ms = (gen_us > rc_gen_us ? gen_us : rc_gen_us) / 1000.0;
    apply_ms = apply_us_max.load() / 1000.0;
    rc_gen_ms = rc_gen_us_max.load() / 1000.0;
    rc_apply_ms = rc_apply_us_max.load() / 1000.0;
}

struct Sample {
    std::vector<llama_token> input_ids;
    int label; // 0 or 1
    llama_token target_token; // verbalizer token for the correct label
};

static std::vector<llama_token> tokenize_text(
    const struct llama_vocab * vocab,
    const std::string & text,
    bool add_special
) {
    int n_tokens = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                                  nullptr, 0, add_special, false);
    if (n_tokens < 0) {
        n_tokens = -n_tokens;
    }
    std::vector<llama_token> tokens(n_tokens);
    int ret = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                             tokens.data(), (int32_t)tokens.size(), add_special, false);
    if (ret < 0) {
        fprintf(stderr, "Error: tokenization failed for text: %.40s...\n", text.c_str());
        tokens.clear();
    } else {
        tokens.resize(ret);
    }
    return tokens;
}

static std::string token_to_string(const struct llama_vocab * vocab, llama_token token) {
    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
    if (n < 0) {
        return "<unk>";
    }
    return std::string(buf, n);
}

static std::vector<Sample> load_sst2_tsv(
    const char * path,
    const struct llama_vocab * vocab,
    int max_samples,
    int max_length,
    std::vector<llama_token> & cls_idx_out
) {
    std::vector<Sample> samples;
    std::ifstream fin(path);
    if (!fin.is_open()) {
        fprintf(stderr, "Error: cannot open TSV file '%s'\n", path);
        return samples;
    }

    std::string line;
    if (!std::getline(fin, line)) {
        fprintf(stderr, "Error: empty TSV file '%s'\n", path);
        return samples;
    }

    if (cls_idx_out.empty()) {
        std::vector<llama_token> t0 = tokenize_text(vocab, "It was terrible", false);
        std::vector<llama_token> t1 = tokenize_text(vocab, "It was great", false);
        if (!t0.empty()) cls_idx_out.push_back(t0.back());
        if (!t1.empty()) cls_idx_out.push_back(t1.back());
    }

    while (std::getline(fin, line) && (int)samples.size() < max_samples) {
        auto tab_pos = line.rfind('\t');
        if (tab_pos == std::string::npos) continue;

        std::string sentence = line.substr(0, tab_pos);
        int label = std::atoi(line.substr(tab_pos + 1).c_str());

        while (!sentence.empty() && (sentence.back() == ' ' || sentence.back() == '\r' || sentence.back() == '\n')) {
            sentence.pop_back();
        }
        while (!sentence.empty() && sentence.front() == ' ') {
            sentence.erase(sentence.begin());
        }

        if (label < 0 || label > 1) continue;

        std::string prompt = sentence + " It was";
        std::vector<llama_token> input_ids = tokenize_text(vocab, prompt, true);

        if ((int)input_ids.size() > max_length) {
            input_ids.resize(max_length);
        }

        Sample s;
        s.input_ids = input_ids;
        s.label = label;
        s.target_token = cls_idx_out[label];
        samples.push_back(s);
    }

    return samples;
}

static float evaluate(
    struct llama_context * ctx,
    const struct llama_vocab * vocab,
    const std::vector<Sample> & eval_samples,
    const std::vector<llama_token> & cls_idx,
    int batch_size,
    int n_vocab
) {
    int correct = 0;
    int total = 0;

    for (size_t si = 0; si < eval_samples.size(); ++si) {
        const auto & sample = eval_samples[si];
        int n_tokens = (int)sample.input_ids.size();

        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        batch.n_tokens = n_tokens;

        for (int i = 0; i < n_tokens; ++i) {
            batch.token[i]    = sample.input_ids[i];
            batch.pos[i]      = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0]= 0;
            batch.logits[i]   = (i == n_tokens - 1) ? 1 : 0;
        }

        ctx->memory->clear(true);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Eval decode failed at sample %zu\n", si);
            llama_batch_free(batch);
            continue;
        }

        float * logits = llama_get_logits_ith(ctx, n_tokens - 1);
        if (!logits) {
            llama_batch_free(batch);
            continue;
        }

        int best_cls = -1;
        float best_logit = -1e30f;
        for (size_t ci = 0; ci < cls_idx.size(); ++ci) {
            llama_token tok = cls_idx[ci];
            if (tok >= 0 && tok < n_vocab && logits[tok] > best_logit) {
                best_logit = logits[tok];
                best_cls = (int)ci;
            }
        }

        if (best_cls == sample.label) {
            correct++;
        }
        total++;

        llama_batch_free(batch);
    }

    float acc = (total > 0) ? (float)correct / total : 0.0f;
    return acc;
}

int main(int argc, char ** argv) {
    const char * model_path    = nullptr;
    const char * train_tsv     = nullptr;
    const char * eval_tsv      = nullptr;
    int          batch_size    = 4;
    int          n_steps       = 1000;
    float        learning_rate = 1e-7f;
    float        sigma         = 0.001f;
    int          max_length    = 128;
    int          eval_interval = 500;
    int          n_threads     = 5;

    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model.gguf> <train.tsv> <eval.tsv> [options]\n", argv[0]);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --batch_size N      (default: %d)\n", batch_size);
        fprintf(stderr, "  --n_steps N         (default: %d)\n", n_steps);
        fprintf(stderr, "  --lr F              (default: %.1e)\n", learning_rate);
        fprintf(stderr, "  --sigma F           (default: %.4f)\n", sigma);
        fprintf(stderr, "  --max_length N      (default: %d)\n", max_length);
        fprintf(stderr, "  --eval_interval N   (default: %d)\n", eval_interval);
        fprintf(stderr, "  --n_threads N       (default: %d)\n", n_threads);
        return 1;
    }

    model_path = argv[1];
    train_tsv  = argv[2];
    eval_tsv   = argv[3];

    for (int i = 4; i < argc; i += 2) {
        if (i + 1 >= argc) break;
        std::string key = argv[i];
        if      (key == "--batch_size")    batch_size    = std::atoi(argv[i+1]);
        else if (key == "--n_steps")       n_steps       = std::atoi(argv[i+1]);
        else if (key == "--lr")            learning_rate = std::atof(argv[i+1]);
        else if (key == "--sigma")         sigma         = (float)std::atof(argv[i+1]);
        else if (key == "--max_length")    max_length    = std::atoi(argv[i+1]);
        else if (key == "--eval_interval") eval_interval = std::atoi(argv[i+1]);
        else if (key == "--n_threads")     n_threads     = std::atoi(argv[i+1]);
    }

    printf("[ZOO-SST2] Config:\n");
    printf("  model_path:    %s\n", model_path);
    printf("  train_tsv:     %s\n", train_tsv);
    printf("  eval_tsv:      %s\n", eval_tsv);
    printf("  batch_size:    %d\n", batch_size);
    printf("  n_steps:       %d\n", n_steps);
    printf("  lr:            %.2e\n", learning_rate);
    printf("  sigma:         %.4f\n", sigma);
    printf("  max_length:    %d\n", max_length);
    printf("  eval_interval: %d\n", eval_interval);
    printf("  n_threads:     %d\n", n_threads);

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = false;

    printf("[ZOO-SST2] Loading model: %s\n", model_path);
    struct llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (model == NULL) {
        fprintf(stderr, "Error: failed to load model '%s'\n", model_path);
        return 1;
    }

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    printf("[ZOO-SST2] Vocab size: %d\n", n_vocab);
    printf("[ZOO-SST2] BOS token: %d, add_bos: %d\n",
           llama_vocab_bos(vocab), llama_vocab_get_add_bos(vocab));

    std::vector<llama_token> cls_idx;

    printf("[ZOO-SST2] Loading training data from: %s\n", train_tsv);
    auto train_data = load_sst2_tsv(train_tsv, vocab, 1000, max_length, cls_idx);
    printf("[ZOO-SST2] Training samples loaded: %zu\n", train_data.size());

    printf("[ZOO-SST2] Loading evaluation data from: %s\n", eval_tsv);
    auto eval_data = load_sst2_tsv(eval_tsv, vocab, 1000, max_length, cls_idx);
    printf("[ZOO-SST2] Evaluation samples loaded: %zu\n", eval_data.size());

    if (train_data.empty() || eval_data.empty()) {
        fprintf(stderr, "Error: failed to load data\n");
        return 1;
    }

    printf("[ZOO-SST2] Classification token IDs (%zu):\n", cls_idx.size());
    for (size_t i = 0; i < cls_idx.size(); ++i) {
        std::string piece = token_to_string(vocab, cls_idx[i]);
        printf("  cls_idx[%zu] = %d (\"%s\") -> label %zu\n", i, cls_idx[i], piece.c_str(), i);
    }

    if (!train_data.empty()) {
        const auto & s0 = train_data[0];
        printf("[ZOO-SST2] Sample 0: label=%d, target_token=%d, n_tokens=%zu\n",
               s0.label, s0.target_token, s0.input_ids.size());
        printf("  Tokens: ");
        for (size_t i = 0; i < std::min((size_t)20, s0.input_ids.size()); ++i) {
            printf("%d ", s0.input_ids[i]);
        }
        if (s0.input_ids.size() > 20) printf("...");
        printf("\n");
    }

    int total_tokens = max_length * batch_size;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = total_tokens;
    ctx_params.n_batch         = total_tokens;
    ctx_params.n_seq_max       = batch_size;
    ctx_params.n_threads       = n_threads;
    ctx_params.n_threads_batch = n_threads;

    struct llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr, "Error: failed to create context\n");
        return 1;
    }
    printf("[ZOO-SST2] System Info: %s\n", llama_print_system_info());

    std::vector<LayerInfo> target_layers;
    std::hash<std::string> hasher;

    for (const auto & kv : model->tensors_by_name) {
        std::string name = kv.first;
        struct ggml_tensor * t = kv.second;
        bool is_matrix = (t->ne[1] > 1);
        bool is_2d = (t->ne[1] > 1 && t->ne[2] == 1 && t->ne[3] == 1);

        if (name.length() > 7 &&
            name.substr(name.length() - 7) == ".weight" &&
            is_matrix) {

            if (t->type != GGML_TYPE_F16) {
                if (!ensure_tensor_f16(t)) {
                    fprintf(stderr, "Warning: tensor '%s' type %d not converted to F16\n", name.c_str(), (int)t->type);
                    continue;
                }
            }

            LayerInfo info;
            info.name = name;
            info.tensor = t;
            info.count = ggml_nelements(t);
            info.name_hash = (uint64_t)hasher(name);
            if (is_2d) {
                info.use_rc = true;
                info.n_rows = t->ne[1];
                info.n_cols = t->ne[0];
            }

            target_layers.push_back(info);
        }
    }
    printf("[ZOO-SST2] Layers to optimize: %zu\n", target_layers.size());

    if (target_layers.empty()) {
        fprintf(stderr, "Error: No valid layers found.\n");
        return 1;
    }

    printf("\n[ZOO-SST2] Initial evaluation (before training)...\n");
    float init_acc = evaluate(ctx, vocab, eval_data, cls_idx, 1, n_vocab);
    printf("[ZOO-SST2] Initial accuracy: %.2f%% (%d samples)\n", init_acc * 100.0f, (int)eval_data.size());

    printf("\n[ZOO-SST2] Starting MeZO training for %d steps (batch_size=%d, lr=%.2e, sigma=%.4f)...\n",
           n_steps, batch_size, learning_rate, sigma);

    std::mt19937 data_rng(42);
    std::vector<int> data_indices(train_data.size());
    std::iota(data_indices.begin(), data_indices.end(), 0);
    int data_cursor = 0;

    double total_loss_sum = 0.0;
    int    total_loss_cnt = 0;
    bool   printed_rc_mem = false;

    for (int step = 0; step < n_steps; ++step) {
        StepTimes tm;
        auto t_step_start = Clock::now();

        if (data_cursor + batch_size > (int)data_indices.size()) {
            std::shuffle(data_indices.begin(), data_indices.end(), data_rng);
            data_cursor = 0;
        }

        std::vector<int> batch_indices(data_indices.begin() + data_cursor,
                                       data_indices.begin() + data_cursor + batch_size);
        data_cursor += batch_size;

        int max_seq_len = 0;
        for (int idx : batch_indices) {
            max_seq_len = std::max(max_seq_len, (int)train_data[idx].input_ids.size());
        }

        int total_batch_tokens = max_seq_len * batch_size;
        llama_token pad_token = llama_vocab_eos(vocab);

        llama_batch batch = llama_batch_init(total_batch_tokens, 0, 1);
        batch.n_tokens = total_batch_tokens;

        std::vector<int> last_token_indices(batch_size);
        std::vector<llama_token> sample_target_tokens(batch_size);

        int token_idx = 0;
        for (int b = 0; b < batch_size; ++b) {
            const auto & sample = train_data[batch_indices[b]];
            int seq_len = (int)sample.input_ids.size();
            sample_target_tokens[b] = sample.target_token;

            for (int i = 0; i < max_seq_len; ++i) {
                if (i < seq_len) {
                    batch.token[token_idx] = sample.input_ids[i];
                } else {
                    batch.token[token_idx] = pad_token;
                }
                batch.pos[token_idx]      = i;
                batch.n_seq_id[token_idx] = 1;
                batch.seq_id[token_idx][0]= b;
                
                batch.logits[token_idx] = (i == seq_len - 1) ? 1 : 0;
                if (i == seq_len - 1) {
                    last_token_indices[b] = token_idx;
                }
                token_idx++;
            }
        }

        uint64_t step_seed = (uint64_t)step * 19937ULL + 12345ULL;

        double pos_gen_ms = 0.0, pos_apply_ms = 0.0, pos_rc_gen_ms = 0.0, pos_rc_apply_ms = 0.0;
        auto t_start = Clock::now();
        apply_perturbation_in_place_parallel(target_layers, step_seed, sigma,
            pos_gen_ms, pos_apply_ms, true, pos_rc_gen_ms, pos_rc_apply_ms);
        tm.t_pos_total_ms = ms_since(t_start, Clock::now());
        tm.t_pos_gen_ms = std::max(pos_gen_ms, pos_rc_gen_ms);
        tm.t_pos_apply_ms = pos_apply_ms;
        tm.t_pos_rc_gen_ms = pos_rc_gen_ms;
        tm.t_pos_rc_apply_ms = pos_rc_apply_ms;

        if (!printed_rc_mem) {
            size_t total_bytes = 0;
            for (const auto & layer : target_layers) {
                if (layer.use_rc && layer.has_rc_cache) {
                    total_bytes += layer.row_noise.size() * sizeof(ggml_fp16_t);
                    total_bytes += layer.col_noise.size() * sizeof(ggml_fp16_t);
                }
            }
            printf("[TIME] RC noise cache size: %.3f MiB\n", total_bytes / 1024.0 / 1024.0);
            printed_rc_mem = true;
        }

        t_start = Clock::now();
        ctx->memory->clear(true);
        tm.t_clear_pos_ms = ms_since(t_start, Clock::now());

        t_start = Clock::now();
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Decode failed at step %d (pos)\n", step);
            llama_batch_free(batch);
            break;
        }
        tm.t_decode_pos_ms = ms_since(t_start, Clock::now());

        float loss_pos = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            float * logits = llama_get_logits_ith(ctx, last_token_indices[b]);
            if (logits) {
                float max_logit = -1e30f;
                for (int i = 0; i < n_vocab; ++i) {
                    if (logits[i] > max_logit) max_logit = logits[i];
                }
                float sum_exp = 0.0f;
                for (int i = 0; i < n_vocab; ++i) {
                    sum_exp += std::exp(logits[i] - max_logit);
                }
                float log_prob = (logits[sample_target_tokens[b]] - max_logit) - std::log(sum_exp);
                loss_pos += -log_prob;
            }
        }
        loss_pos /= batch_size;

        double neg_gen_ms = 0.0, neg_apply_ms = 0.0, neg_rc_gen_ms = 0.0, neg_rc_apply_ms = 0.0;
        t_start = Clock::now();
        apply_perturbation_in_place_parallel(target_layers, step_seed, -2.0f * sigma,
            neg_gen_ms, neg_apply_ms, false, neg_rc_gen_ms, neg_rc_apply_ms);
        tm.t_neg_total_ms = ms_since(t_start, Clock::now());
        tm.t_neg_gen_ms = neg_gen_ms;
        tm.t_neg_apply_ms = neg_apply_ms;

        t_start = Clock::now();
        ctx->memory->clear(true);
        tm.t_clear_neg_ms = ms_since(t_start, Clock::now());

        t_start = Clock::now();
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Decode failed at step %d (neg)\n", step);
            llama_batch_free(batch);
            break;
        }
        tm.t_decode_neg_ms = ms_since(t_start, Clock::now());

        float loss_neg = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            float * logits = llama_get_logits_ith(ctx, last_token_indices[b]);
            if (logits) {
                float max_logit = -1e30f;
                for (int i = 0; i < n_vocab; ++i) {
                    if (logits[i] > max_logit) max_logit = logits[i];
                }
                float sum_exp = 0.0f;
                for (int i = 0; i < n_vocab; ++i) {
                    sum_exp += std::exp(logits[i] - max_logit);
                }
                float log_prob = (logits[sample_target_tokens[b]] - max_logit) - std::log(sum_exp);
                loss_neg += -log_prob;
            }
        }
        loss_neg /= batch_size;

        float loss_diff = loss_pos - loss_neg;
        float grad_est = loss_diff / (2.0f * sigma);
        float update_scale = sigma - (learning_rate * grad_est);

        double upd_gen_ms = 0.0, upd_apply_ms = 0.0, upd_rc_gen_ms = 0.0, upd_rc_apply_ms = 0.0;
        t_start = Clock::now();
        apply_perturbation_in_place_parallel(target_layers, step_seed, update_scale,
            upd_gen_ms, upd_apply_ms, false, upd_rc_gen_ms, upd_rc_apply_ms);
        tm.t_update_total_ms = ms_since(t_start, Clock::now());
        tm.t_update_gen_ms = upd_gen_ms;
        tm.t_update_apply_ms = upd_apply_ms;

        auto t_step_end = Clock::now();
        tm.t_step_total_ms = ms_since(t_step_start, t_step_end);

        float avg_loss = (loss_pos + loss_neg) / 2.0f;
        total_loss_sum += avg_loss;
        total_loss_cnt += 1;

        printf("Step %d/%d | L+: %.4f | L-: %.4f | Grad: %.6f | AvgLoss: %.4f | step: %.1f ms\n",
               step, n_steps, loss_pos, loss_neg, grad_est, avg_loss, tm.t_step_total_ms);

        llama_batch_free(batch);

        if ((step + 1) % eval_interval == 0) {
            printf("\n[ZOO-SST2] Evaluation at step %d...\n", step + 1);
            float acc = evaluate(ctx, vocab, eval_data, cls_idx, 1, n_vocab);
            printf("[ZOO-SST2] Step %d accuracy: %.2f%% | RunningAvgLoss: %.4f\n\n",
                   step + 1, acc * 100.0f,
                   total_loss_cnt > 0 ? total_loss_sum / total_loss_cnt : 0.0);
        }
    }

    printf("\n[ZOO-SST2] Final evaluation...\n");
    float final_acc = evaluate(ctx, vocab, eval_data, cls_idx, 1, n_vocab);
    printf("[ZOO-SST2] Final accuracy: %.2f%% (%d samples)\n", final_acc * 100.0f, (int)eval_data.size());
    printf("[ZOO-SST2] Average training loss: %.4f\n",
           total_loss_cnt > 0 ? total_loss_sum / total_loss_cnt : 0.0);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}