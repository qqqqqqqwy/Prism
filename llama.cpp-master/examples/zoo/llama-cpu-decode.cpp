#include "llama.h"
#include "ggml.h"

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

int main(int argc, char ** argv) {
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = false;

    const char* model_path = "/data/local/tmp/TinyLlama-1.1B-Chat-v1.0.FP16.gguf";
    if (argc > 1) model_path = argv[1];

    printf("[INFO] Loading model: %s\n", model_path);
    struct llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "Error: failed to load model\n");
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    ctx_params.n_threads = 5;
    ctx_params.n_threads_batch = 5;

    struct llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Error: failed to create context\n");
        return 1;
    }

    printf("[INFO] System Info: %s\n", llama_print_system_info());

    // ===== 构造固定输入 =====
    std::vector<llama_token> tokens;
    for (int i = 0; i < 30; ++i) {
        tokens.push_back(1 + i);
    }

    int n_tokens = tokens.size();

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;

    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1);
    }

    printf("\n[INFO] Starting Benchmark: Prefill vs Decoding...\n");

    // 1. 先进行一次 Prefill (处理前 30 个 token)
    auto t_pre_start = std::chrono::steady_clock::now();
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Prefill failed\n");
        return 1;
    }
    auto t_pre_end = std::chrono::steady_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_pre_end - t_pre_start).count();
    printf("[RESULT] Prefill (30 tokens) time: %.3f ms (%.3f ms/tok)\n", prefill_ms, prefill_ms / 30.0);

    // 2. 开始测试单 Token 解码 (Decoding)
    int n_decode = 20; // 测试生成 20 个 token
    double total_decode_ms = 0.0;
    llama_token last_token = 1; // 模拟生成的上一个 token

    for (int i = 0; i < n_decode; ++i) {
        // 构造一个只包含 1 个 token 的 batch
        llama_batch s_batch = llama_batch_init(1, 0, 1);
        s_batch.n_tokens = 1;
        s_batch.token[0]    = last_token;
        s_batch.pos[0]      = n_tokens + i; // 位置接在 30 之后
        s_batch.n_seq_id[0] = 1;
        s_batch.seq_id[0][0] = 0;
        s_batch.logits[0]   = true;

        auto t_dec_start = std::chrono::steady_clock::now();
        
        // 执行解码
        if (llama_decode(ctx, s_batch) != 0) {
            printf("Decode failed at step %d\n", i);
            break;
        }
        
        auto t_dec_end = std::chrono::steady_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(t_dec_end - t_dec_start).count();
        
        total_decode_ms += step_ms;
        printf("  Token %d | Decode time: %.3f ms\n", i + 1, step_ms);
        
        llama_batch_free(s_batch);
    }

    printf("\n[RESULT] Average Single Token Decode time: %.3f ms\n", total_decode_ms / n_decode);
    printf("[RESULT] Generation speed: %.2f tokens/s\n", 1000.0 / (total_decode_ms / n_decode));

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}