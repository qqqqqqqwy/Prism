#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <vector>
#include <string>
#include <random>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <functional>
#include <thread>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <cctype>

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

// Fast RNG: xoshiro256++ with splitmix64 seeding.
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

// Ziggurat normal (128 segments) initialization.
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

struct StaticLayerPlan {
    std::unordered_set<std::string> int8_layers;
    std::unordered_set<std::string> int4_layers;
    std::unordered_set<std::string> all_layers;
};

static bool has_suffix(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool is_weight_matrix(const std::string & name, const ggml_tensor * t) {
    return has_suffix(name, ".weight") && t->ne[1] > 1;
}

static std::string to_lower(std::string s) {
    for (char & c : s) {
        c = (char)std::tolower((unsigned char)c);
    }
    return s;
}

static std::string regex_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() * 2);
    const std::string meta = R"(\.^$|()[]{}*+?)";
    for (char c : s) {
        if (meta.find(c) != std::string::npos) {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    return out;
}

static int extract_block_id(const std::string & name) {
    const char * tags[] = {"blk.", "layers."};
    for (const char * tag : tags) {
        const size_t pos = name.find(tag);
        if (pos == std::string::npos) {
            continue;
        }
        size_t i = pos + std::strlen(tag);
        if (i >= name.size() || !std::isdigit((unsigned char)name[i])) {
            continue;
        }
        int value = 0;
        while (i < name.size() && std::isdigit((unsigned char)name[i])) {
            value = value * 10 + (name[i] - '0');
            ++i;
        }
        return value;
    }
    return -1;
}

static std::vector<std::string> collect_weight_names(struct llama_model * model) {
    std::vector<std::string> names;
    names.reserve(model->tensors_by_name.size());
    for (const auto & kv : model->tensors_by_name) {
        if (is_weight_matrix(kv.first, kv.second)) {
            names.push_back(kv.first);
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}

static StaticLayerPlan build_static_layer_plan(
    const std::vector<std::string> & weight_names,
    int n_int8_blocks,
    int n_int4_blocks
) {
    StaticLayerPlan plan;
    std::map<int, std::vector<std::string>> by_block;
    std::vector<std::string> no_block;

    for (const std::string & name : weight_names) {
        const int bid = extract_block_id(name);
        if (bid >= 0) {
            by_block[bid].push_back(name);
        } else {
            no_block.push_back(name);
        }
    }

    int i = 0;
    for (const auto & kv : by_block) {
        const bool pick_int8 = i < n_int8_blocks;
        const bool pick_int4 = !pick_int8 && i < (n_int8_blocks + n_int4_blocks);
        if (!pick_int8 && !pick_int4) {
            ++i;
            continue;
        }
        for (const std::string & name : kv.second) {
            if (pick_int8) {
                plan.int8_layers.insert(name);
            } else {
                plan.int4_layers.insert(name);
            }
            plan.all_layers.insert(name);
        }
        ++i;
    }

    // Backfill from non-block tensors if model naming differs.
    size_t target_min = (size_t)std::max(0, n_int8_blocks + n_int4_blocks);
    if (plan.all_layers.size() < target_min) {
        for (const std::string & name : no_block) {
            if (plan.all_layers.find(name) != plan.all_layers.end()) {
                continue;
            }
            if (plan.int8_layers.size() < (size_t)std::max(0, n_int8_blocks)) {
                plan.int8_layers.insert(name);
            } else if (plan.int4_layers.size() < (size_t)std::max(0, n_int4_blocks)) {
                plan.int4_layers.insert(name);
            } else {
                break;
            }
            plan.all_layers.insert(name);
        }
    }

    return plan;
}

static ggml_backend_buffer_type_t find_npu_buffer_type(std::string & chosen_device_name, std::string & chosen_buft_name) {
    ggml_backend_buffer_type_t chosen = nullptr;
    const size_t ndev = ggml_backend_dev_count();

    for (size_t i = 0; i < ndev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev == nullptr) {
            continue;
        }
        const char * dev_name_c = ggml_backend_dev_name(dev);
        const char * dev_desc_c = ggml_backend_dev_description(dev);
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
        const char * buft_name_c = buft ? ggml_backend_buft_name(buft) : "";
        const std::string dev_name = dev_name_c ? dev_name_c : "";
        const std::string dev_desc = dev_desc_c ? dev_desc_c : "";
        const std::string buft_name = buft_name_c ? buft_name_c : "";

        const std::string key = to_lower(dev_name + " " + dev_desc + " " + buft_name);
        const bool looks_npu = key.find("npu") != std::string::npos ||
                               key.find("htp") != std::string::npos ||
                               key.find("qnn") != std::string::npos;
        if (looks_npu && buft != nullptr) {
            chosen = buft;
            chosen_device_name = dev_name;
            chosen_buft_name = buft_name;
            break;
        }
    }

    return chosen;
}

void apply_perturbation_in_place_parallel(
    std::vector<LayerInfo>& layers,
    uint64_t step_seed,
    float scale,
    float std_dev,
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
                            float n = ZigguratNormal::sample(rng) * std_dev;
                            layer.row_noise[r] = fp32_to_fp16(n);
                        }
                        for (int64_t c = 0; c < layer.n_cols; ++c) {
                            float n = ZigguratNormal::sample(rng) * std_dev;
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
                        float n = ZigguratNormal::sample(rng) * std_dev * scale;
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

int main(int argc, char ** argv) {
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    ggml_backend_load_all();

    const char* model_path = "/data/local/tmp/TinyLlama-1.1B-Chat-v1.0.FP16.gguf";
    if (argc > 1) model_path = argv[1];

    int n_int8_blocks = 15;
    int n_int4_blocks = 10;
    if (argc > 2) n_int8_blocks = std::max(0, std::atoi(argv[2]));
    if (argc > 3) n_int4_blocks = std::max(0, std::atoi(argv[3]));
    printf("[ZOO] Args: <model_path> [int8_blocks=%d] [int4_blocks=%d]\n", n_int8_blocks, n_int4_blocks);

    std::string npu_dev_name;
    std::string npu_buft_name;
    ggml_backend_buffer_type_t npu_buft = find_npu_buffer_type(npu_dev_name, npu_buft_name);

    printf("[ZOO] Model: %s\n", model_path);
    if (npu_buft) {
        printf("[ZOO] NPU buffer selected: device='%s' buft='%s'\n", npu_dev_name.c_str(), npu_buft_name.c_str());
    } else {
        printf("[ZOO] NPU backend not detected. Fallback to CPU-only placement.\n");
    }

    llama_model_params scan_params = llama_model_default_params();
    scan_params.use_mmap = false;
    scan_params.no_alloc = true;
    struct llama_model * scan_model = llama_model_load_from_file(model_path, scan_params);
    if (scan_model == NULL) {
        fprintf(stderr, "Error: failed pre-scan load for model '%s'\n", model_path);
        llama_backend_free();
        return 1;
    }

    const std::vector<std::string> weight_names = collect_weight_names(scan_model);
    const StaticLayerPlan static_plan = build_static_layer_plan(weight_names, n_int8_blocks, n_int4_blocks);
    llama_model_free(scan_model);

    std::vector<std::string> override_patterns;
    std::vector<llama_model_tensor_buft_override> overrides;
    if (npu_buft && !static_plan.all_layers.empty()) {
        override_patterns.reserve(static_plan.all_layers.size());
        overrides.reserve(static_plan.all_layers.size() + 1);
        for (const std::string & name : static_plan.all_layers) {
            override_patterns.push_back("^" + regex_escape(name) + "$");
        }
        for (const std::string & pat : override_patterns) {
            llama_model_tensor_buft_override item;
            item.pattern = pat.c_str();
            item.buft = npu_buft;
            overrides.push_back(item);
        }
        overrides.push_back({nullptr, nullptr});
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = false;
    if (!overrides.empty()) {
        model_params.tensor_buft_overrides = overrides.data();
    }

    printf("[ZOO] Loading model with placement overrides...\n");
    struct llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (model == NULL) {
        fprintf(stderr, "Error: failed to load model '%s'\n", model_path);
        llama_backend_free();
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;   
    ctx_params.n_batch = 512; 
    ctx_params.n_threads = 5;
    ctx_params.n_threads_batch = 5;
    
    struct llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr, "Error: failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    printf("[ZOO] System Info: %s\n", llama_print_system_info());

    std::vector<LayerInfo> target_layers;
    size_t static_int8_loaded = 0;
    size_t static_int4_loaded = 0;
    std::hash<std::string> hasher;

    for (const auto & kv : model->tensors_by_name) {
        const std::string & name = kv.first;
        struct ggml_tensor * t = kv.second;
        if (!is_weight_matrix(name, t)) {
            continue;
        }

        if (static_plan.int8_layers.find(name) != static_plan.int8_layers.end()) {
            static_int8_loaded++;
            continue;
        }
        if (static_plan.int4_layers.find(name) != static_plan.int4_layers.end()) {
            static_int4_loaded++;
            continue;
        }

        const bool is_2d = (t->ne[1] > 1 && t->ne[2] == 1 && t->ne[3] == 1);

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
    printf("[ZOO] Static NPU layers (target): int8=%zu int4=%zu\n",
           static_plan.int8_layers.size(), static_plan.int4_layers.size());
    printf("[ZOO] Static NPU layers (loaded): int8=%zu int4=%zu\n", static_int8_loaded, static_int4_loaded);
    printf("[ZOO] Trainable CPU layers: %zu\n", target_layers.size());

    if (target_layers.empty()) {
        fprintf(stderr, "Error: no trainable CPU layers found after static placement.\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::vector<llama_token> tokens = { 1, 100, 200, 300, 400, 500 }; 
    tokens.clear();
    for (int i = 0; i < 30; ++i) {
        tokens.push_back(1 + i);
    }
    int n_tokens = tokens.size();
    llama_token target_label_token = 999; 

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0]= 0;
        batch.logits[i] = (i == n_tokens - 1);
    }

    float learning_rate = 0.0000001f;
    float sigma = 0.001f;
    printf("\n[ZOO] Starting Loop (CPU trainable + static NPU layers)...\n");

    double rng_gen_sum_ms = 0.0;
    int rng_gen_count = 0;
    double update_sum_ms = 0.0;
    int update_count = 0;
    double decode_sum_ms = 0.0;
    int decode_count = 0;
    bool printed_rc_mem = false;

    for (int step = 0; step < 5; ++step) {
        StepTimes tm;
        auto t_step_start = Clock::now();
        
        uint64_t step_seed = (uint64_t)step * 19937ULL + 12345ULL;
        double pos_gen_ms = 0.0;
        double pos_apply_ms = 0.0;
        double pos_rc_gen_ms = 0.0;
        double pos_rc_apply_ms = 0.0;
        auto t_start = Clock::now();
        apply_perturbation_in_place_parallel(target_layers, step_seed, 1.0f, sigma, pos_gen_ms, pos_apply_ms, true, pos_rc_gen_ms, pos_rc_apply_ms);
        tm.t_pos_total_ms = ms_since(t_start, Clock::now());
        tm.t_pos_gen_ms = std::max(pos_gen_ms, pos_rc_gen_ms);
        tm.t_pos_apply_ms = pos_apply_ms;
        tm.t_pos_rc_gen_ms = pos_rc_gen_ms;
        tm.t_pos_rc_apply_ms = pos_rc_apply_ms;
        rng_gen_sum_ms += std::max(pos_gen_ms, pos_rc_gen_ms);
        rng_gen_count += 1;
        update_sum_ms += pos_apply_ms;
        update_count += 1;

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
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "Decode failed\n"); break; }
        tm.t_decode_pos_ms = ms_since(t_start, Clock::now());
        decode_sum_ms += tm.t_decode_pos_ms;
        decode_count += 1;
        
        float loss_pos = -llama_get_logits_ith(ctx, n_tokens - 1)[target_label_token];

        double neg_gen_ms = 0.0;
        double neg_apply_ms = 0.0;
        t_start = Clock::now();
        double neg_rc_gen_ms = 0.0;
        double neg_rc_apply_ms = 0.0;
        apply_perturbation_in_place_parallel(target_layers, step_seed, -2.0f, sigma, neg_gen_ms, neg_apply_ms, false, neg_rc_gen_ms, neg_rc_apply_ms);
        tm.t_neg_total_ms = ms_since(t_start, Clock::now());
        tm.t_neg_gen_ms = neg_gen_ms;
        tm.t_neg_apply_ms = neg_apply_ms;
        rng_gen_sum_ms += neg_gen_ms;
        rng_gen_count += 1;
        update_sum_ms += neg_apply_ms;
        update_count += 1;

        t_start = Clock::now();
        ctx->memory->clear(true);
        tm.t_clear_neg_ms = ms_since(t_start, Clock::now());

        t_start = Clock::now();
        if (llama_decode(ctx, batch) != 0) break;
        tm.t_decode_neg_ms = ms_since(t_start, Clock::now());
        decode_sum_ms += tm.t_decode_neg_ms;
        decode_count += 1;

        float loss_neg = -llama_get_logits_ith(ctx, n_tokens - 1)[target_label_token];

        float loss_diff = loss_pos - loss_neg;
        float grad_est = loss_diff / (2.0f * sigma);
        
        float update_scale = 1.0f - (learning_rate * grad_est);

        double upd_gen_ms = 0.0;
        double upd_apply_ms = 0.0;
        t_start = Clock::now();
        double upd_rc_gen_ms = 0.0;
        double upd_rc_apply_ms = 0.0;
        apply_perturbation_in_place_parallel(target_layers, step_seed, update_scale, sigma, upd_gen_ms, upd_apply_ms, false, upd_rc_gen_ms, upd_rc_apply_ms);
        tm.t_update_total_ms = ms_since(t_start, Clock::now());
        tm.t_update_gen_ms = upd_gen_ms;
        tm.t_update_apply_ms = upd_apply_ms;
        rng_gen_sum_ms += upd_gen_ms;
        rng_gen_count += 1;
        update_sum_ms += upd_apply_ms;
        update_count += 1;

        auto t_step_end = Clock::now();
        tm.t_step_total_ms = ms_since(t_step_start, t_step_end);
        
        printf("Step %d | L+: %.4f | L-: %.4f | Grad: %.4f\n", step, loss_pos, loss_neg, grad_est);
        tm.print(step);
    }

    if (rng_gen_count > 0) {
        printf("[TIME] RNG gen avg over %d runs: %.3f ms\n", rng_gen_count, rng_gen_sum_ms / rng_gen_count);
    }
    if (update_count > 0) {
        printf("[TIME] Update apply avg over %d runs: %.3f ms\n", update_count, update_sum_ms / update_count);
    }
    if (decode_count > 0) {
        printf("[TIME] Decode avg over %d runs: %.3f ms\n", decode_count, decode_sum_ms / decode_count);
    }
    if (!npu_buft) {
        printf("[ZOO] NOTE: NPU backend not found, this run did not offload static layers.\n");
    } else if (overrides.empty()) {
        printf("[ZOO] NOTE: NPU backend found, but no static layer overrides were generated.\n");
    } else {
        printf("[ZOO] NOTE: static layer placement applied via tensor_buft_overrides.\n");
        printf("[ZOO] NOTE: int8/int4 here are static assignment groups; per-layer datatype still depends on source GGUF / backend repacking.\n");
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
