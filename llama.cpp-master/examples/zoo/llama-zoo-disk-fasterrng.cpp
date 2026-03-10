#include "llama.h"
#include "ggml.h"

#include <vector>
#include <string>
#include <random>
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <sys/stat.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <future>
#include <cmath>
#include <cstdint>

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
        // (0,1) exclusive
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
    double t_noise_total_ms    = 0.0;
    double t_noise_gen_ms      = 0.0;
    double t_noise_io_ms       = 0.0;

    double t_pos_total_ms      = 0.0;
    double t_pos_read_ms       = 0.0;
    double t_pos_apply_ms      = 0.0;
    double t_perturb_pos_ms    = 0.0;
    double t_clear_pos_ms      = 0.0;
    double t_decode_pos_ms     = 0.0;

    double t_neg_total_ms      = 0.0;
    double t_neg_read_ms       = 0.0;
    double t_neg_apply_ms      = 0.0;
    double t_perturb_neg_ms    = 0.0;
    double t_clear_neg_ms      = 0.0;
    double t_decode_neg_ms     = 0.0;

    double t_update_total_ms   = 0.0;
    double t_update_read_ms    = 0.0;
    double t_update_apply_ms   = 0.0;
    double t_update_ms         = 0.0;
    double t_step_total_ms     = 0.0;

    void print(int step) const {
        printf(
            "[TIME] Step %d | total %.3f ms\n"
            "      noise_total: %.3f ms | gen(max): %.3f ms | io(max): %.3f ms\n"
            "      +perturb:    total %.3f ms | read(max): %.3f ms | apply(max): %.3f ms | +clear: %.3f ms | +decode: %.3f ms\n"
            "      -perturb:    total %.3f ms | read(max): %.3f ms | apply(max): %.3f ms | -clear: %.3f ms | -decode: %.3f ms\n"
            "      update:      total %.3f ms | read(max): %.3f ms | apply(max): %.3f ms\n",
            step, t_step_total_ms,
            t_noise_total_ms, t_noise_gen_ms, t_noise_io_ms,
            t_pos_total_ms, t_pos_read_ms, t_pos_apply_ms, t_clear_pos_ms, t_decode_pos_ms,
            t_neg_total_ms, t_neg_read_ms, t_neg_apply_ms, t_clear_neg_ms, t_decode_neg_ms,
            t_update_total_ms, t_update_read_ms, t_update_apply_ms
        );
    }
};

std::string get_noise_path(const std::string& layer_name) {
    std::string safe_name = layer_name;
    std::replace(safe_name.begin(), safe_name.end(), '.', '_');
    return "/data/local/tmp/zoo_noise_" + safe_name + ".bin";
}

struct LayerInfo {
    std::string name;
    ggml_tensor* tensor;
    int64_t count;
};

void generate_and_save_noise_to_disk_parallel(const std::vector<LayerInfo>& layers, float std_dev, double & gen_ms, double & io_ms) {
    ZigguratNormal::init();

    std::atomic<size_t> layer_idx(0);
    std::vector<std::thread> workers;

    const uint64_t base_seed = (uint64_t) std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::atomic<uint64_t> seed_counter(0);
    std::atomic<int64_t> gen_us_max(0);
    std::atomic<int64_t> io_us_max(0);

    for (int t = 0; t < N_WORKERS; ++t) {
        workers.emplace_back([&]() {
            const uint64_t seed = base_seed + (seed_counter.fetch_add(1) + 1) * 0x9e3779b97f4a7c15ULL;
            Xoshiro256pp rng(seed);
            std::vector<float> local_buffer;
            int64_t local_gen_us = 0;
            int64_t local_io_us = 0;

            while (true) {
                size_t i = layer_idx.fetch_add(1);
                if (i >= layers.size()) break;

                const auto& layer = layers[i];
                if (local_buffer.size() < layer.count) {
                    local_buffer.resize(layer.count);
                }
                auto t0 = Clock::now();
                for (int64_t j = 0; j < layer.count; ++j) {
                    local_buffer[j] = ZigguratNormal::sample(rng) * std_dev;
                }
                auto t1 = Clock::now();
                local_gen_us += (int64_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

                auto t2 = Clock::now();
                std::string path = get_noise_path(layer.name);
                std::ofstream outfile(path, std::ios::binary);
                if (outfile) {
                    outfile.write(reinterpret_cast<const char*>(local_buffer.data()), layer.count * sizeof(float));
                }
                auto t3 = Clock::now();
                local_io_us += (int64_t) std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
            }

            int64_t prev = gen_us_max.load();
            while (prev < local_gen_us && !gen_us_max.compare_exchange_weak(prev, local_gen_us)) {}
            prev = io_us_max.load();
            while (prev < local_io_us && !io_us_max.compare_exchange_weak(prev, local_io_us)) {}
        });
    }

    for (auto& w : workers) w.join();

    gen_ms = gen_us_max.load() / 1000.0;
    io_ms = io_us_max.load() / 1000.0;
}

void apply_noise_scale_parallel(std::vector<LayerInfo>& layers, float scale, double & read_ms, double & apply_ms) {
    if (scale == 0.0f) return;

    std::atomic<size_t> layer_idx(0);
    std::vector<std::thread> workers;
    std::atomic<int64_t> read_us_max(0);
    std::atomic<int64_t> apply_us_max(0);

    for (int t = 0; t < N_WORKERS; ++t) {
        workers.emplace_back([&]() {
            std::vector<float> local_buffer;
            int64_t local_read_us = 0;
            int64_t local_apply_us = 0;

            while (true) {
                size_t i = layer_idx.fetch_add(1);
                if (i >= layers.size()) break;

                auto& layer = layers[i];
                
                std::string path = get_noise_path(layer.name);
                auto t_read0 = Clock::now();
                std::ifstream infile(path, std::ios::binary);
                if (!infile) continue;

                if (local_buffer.size() < layer.count) {
                    local_buffer.resize(layer.count);
                }
                
                infile.read(reinterpret_cast<char*>(local_buffer.data()), layer.count * sizeof(float));
                infile.close();
                auto t_read1 = Clock::now();
                local_read_us += (int64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_read1 - t_read0).count();

                ggml_fp16_t* current_ptr = (ggml_fp16_t*)layer.tensor->data;
                const float* noise_ptr = local_buffer.data();

                auto t_apply0 = Clock::now();
                for (int64_t j = 0; j < layer.count; ++j) {
                    float val = fp16_to_fp32(current_ptr[j]);
                    val += scale * noise_ptr[j];
                    current_ptr[j] = fp32_to_fp16(val);
                }
                auto t_apply1 = Clock::now();
                local_apply_us += (int64_t) std::chrono::duration_cast<std::chrono::microseconds>(t_apply1 - t_apply0).count();
            }

            int64_t prev = read_us_max.load();
            while (prev < local_read_us && !read_us_max.compare_exchange_weak(prev, local_read_us)) {}
            prev = apply_us_max.load();
            while (prev < local_apply_us && !apply_us_max.compare_exchange_weak(prev, local_apply_us)) {}
        });
    }

    for (auto& w : workers) w.join();

    read_ms = read_us_max.load() / 1000.0;
    apply_ms = apply_us_max.load() / 1000.0;
}

int main(int argc, char ** argv) {
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = false; 
    
    const char* model_path = "/data/local/tmp/TinyLlama-1.1B-Chat-v1.0.FP16.gguf";
    if (argc > 1) model_path = argv[1];
    
    printf("[ZOO] Loading model: %s\n", model_path);
    struct llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (model == NULL) {
        fprintf(stderr, "Error: failed to load model '%s'\n", model_path);
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
        return 1;
    }
    printf("[ZOO] System Info: %s\n", llama_print_system_info());

    std::vector<LayerInfo> target_layers;
    
    for (const auto & kv : model->tensors_by_name) {
        std::string name = kv.first;
        struct ggml_tensor * t = kv.second;
        bool is_matrix = (t->ne[1] > 1);

        if (name.length() > 7 && 
            name.substr(name.length() - 7) == ".weight" && 
            is_matrix) {
            
            LayerInfo info;
            info.name = name;
            info.tensor = t;
            info.count = ggml_nelements(t);
            
            target_layers.push_back(info);
        }
    }
    printf("[ZOO] Layers: %zu\n", target_layers.size());

    std::vector<llama_token> tokens = { 1, 100, 200, 300, 400, 500 }; 
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
    printf("\n[ZOO] Starting Loop...\n");

    for (int step = 0; step < 5; ++step) {
        StepTimes tm;
        auto t_step_start = Clock::now();
        auto t_temp_start = Clock::now();
        auto t_temp_end = Clock::now();

        t_temp_start = Clock::now();
        double noise_gen_ms = 0.0;
        double noise_io_ms = 0.0;
        generate_and_save_noise_to_disk_parallel(target_layers, sigma, noise_gen_ms, noise_io_ms);
        t_temp_end = Clock::now();
        tm.t_noise_total_ms = ms_since(t_temp_start, t_temp_end);
        tm.t_noise_gen_ms = noise_gen_ms;
        tm.t_noise_io_ms = noise_io_ms;

        double pos_read_ms = 0.0;
        double pos_apply_ms = 0.0;
        t_temp_start = Clock::now();
        apply_noise_scale_parallel(target_layers, 1.0f, pos_read_ms, pos_apply_ms);
        t_temp_end = Clock::now();
        tm.t_pos_total_ms = ms_since(t_temp_start, t_temp_end);
        tm.t_pos_read_ms = pos_read_ms;
        tm.t_pos_apply_ms = pos_apply_ms;
        tm.t_perturb_pos_ms = tm.t_pos_total_ms;

        t_temp_start = Clock::now();
        ctx->memory->clear(true);
        t_temp_end = Clock::now();
        tm.t_clear_pos_ms = ms_since(t_temp_start, t_temp_end);

        t_temp_start = Clock::now();
        if (llama_decode(ctx, batch) != 0) break;
        t_temp_end = Clock::now();
        tm.t_decode_pos_ms = ms_since(t_temp_start, t_temp_end);
        
        float loss_pos = -llama_get_logits_ith(ctx, n_tokens - 1)[target_label_token];

        double neg_read_ms = 0.0;
        double neg_apply_ms = 0.0;
        t_temp_start = Clock::now();
        apply_noise_scale_parallel(target_layers, -2.0f, neg_read_ms, neg_apply_ms);
        t_temp_end = Clock::now();
        tm.t_neg_total_ms = ms_since(t_temp_start, t_temp_end);
        tm.t_neg_read_ms = neg_read_ms;
        tm.t_neg_apply_ms = neg_apply_ms;
        tm.t_perturb_neg_ms = tm.t_neg_total_ms;

        t_temp_start = Clock::now();
        ctx->memory->clear(true);
        t_temp_end = Clock::now();
        tm.t_clear_neg_ms = ms_since(t_temp_start, t_temp_end);

        t_temp_start = Clock::now();
        if (llama_decode(ctx, batch) != 0) break;
        t_temp_end = Clock::now();
        tm.t_decode_neg_ms = ms_since(t_temp_start, t_temp_end);

        float loss_neg = -llama_get_logits_ith(ctx, n_tokens - 1)[target_label_token];

        float loss_diff = loss_pos - loss_neg;
        float grad_est = loss_diff / (2.0f * sigma);
        
        double upd_read_ms = 0.0;
        double upd_apply_ms = 0.0;
        t_temp_start = Clock::now();
        float restore_and_update_scale = 1.0f - (learning_rate * grad_est);
        apply_noise_scale_parallel(target_layers, restore_and_update_scale, upd_read_ms, upd_apply_ms);
        t_temp_end = Clock::now();
        tm.t_update_total_ms = ms_since(t_temp_start, t_temp_end);
        tm.t_update_read_ms = upd_read_ms;
        tm.t_update_apply_ms = upd_apply_ms;
        tm.t_update_ms = tm.t_update_total_ms;

        auto t_step_end = Clock::now();
        tm.t_step_total_ms = ms_since(t_step_start, t_step_end);
        
        printf("Step %d | L+: %.4f | L-: %.4f | Grad: %.4f\n", step, loss_pos, loss_neg, grad_est);
        
        tm.print(step);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
