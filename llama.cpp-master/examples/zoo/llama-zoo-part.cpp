#include "llama.h"
#include "ggml.h"

#include <vector>
#include <string>
#include <random>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <chrono>

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
    double t_noise_gen_ms      = 0.0;

    double t_perturb_pos_ms    = 0.0;
    double t_clear_pos_ms      = 0.0;
    double t_decode_pos_ms     = 0.0;

    double t_perturb_neg_ms    = 0.0;
    double t_clear_neg_ms      = 0.0;
    double t_decode_neg_ms     = 0.0;

    double t_update_ms         = 0.0;
    double t_step_total_ms     = 0.0;

    void print(int step) const {
        printf(
            "[TIME] Step %d | total %.3f ms\n"
            "       noise_gen:   %.3f ms\n"
            "       +perturb:    %.3f ms | +clear: %.3f ms | +decode: %.3f ms\n"
            "       -perturb:    %.3f ms | -clear: %.3f ms | -decode: %.3f ms\n"
            "       update:      %.3f ms\n",
            step, t_step_total_ms,
            t_noise_gen_ms,
            t_perturb_pos_ms, t_clear_pos_ms, t_decode_pos_ms,
            t_perturb_neg_ms, t_clear_neg_ms, t_decode_neg_ms,
            t_update_ms
        );
    }
};

// 生成高斯噪声
std::vector<float> generate_gaussian_noise(int size, float std_dev = 0.02f) {
    std::vector<float> noise(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0.0f, std_dev);
    for (int i = 0; i < size; ++i) {
        noise[i] = d(gen);
    }
    return noise;
}

int main(int argc, char ** argv) {
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = false;
    
    const char* model_path = "/data/local/tmp/TinyLlama-1.1B-Chat-v1.0.FP16.gguf";
    if (argc > 1) {
        model_path = argv[1];
    }
    
    printf("[ZOO] Loading model from: %s\n", model_path);
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

    std::string target_layer_name = "blk.0.attn_q.weight";
    struct ggml_tensor * target_tensor = nullptr;

    for (const auto & kv : model->tensors_by_name) {
        if (kv.first == target_layer_name) {
            target_tensor = kv.second;
            break;
        }
    }

    if (!target_tensor) {
        fprintf(stderr, "Error: Tensor '%s' not found!\n", target_layer_name.c_str());
        printf("Available tensors (first 5):\n");
        for(size_t i=0; i<5 && i<model->tensors_by_name.size(); ++i) {
            printf(" - %s\n", model->tensors_by_name[i].first.c_str());
        }
        return 1;
    }

    int64_t num_elements = ggml_nelements(target_tensor);
    ggml_fp16_t* weight_ptr = (ggml_fp16_t*)target_tensor->data;

    printf("[ZOO] Target Found: %s\n", target_layer_name.c_str());
    printf("[ZOO] Elements: %ld\n", num_elements);

    std::vector<llama_token> tokens = { 1, 100, 200, 300, 400, 500 }; 
    int n_tokens = tokens.size();
    llama_token target_label_token = 999; 

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0]= 0;
        batch.logits[i]   = (i == n_tokens - 1);
    }
    // 7. 零阶优化循环
    std::vector<ggml_fp16_t> base_weights(weight_ptr, weight_ptr + num_elements);
    float learning_rate = 0.001f;
    float sigma = 0.02f;
    printf("\n[ZOO] Starting Loop...\n");

    for (int step = 0; step < 5; ++step) {
        StepTimes tm;
        auto t_step0 = Clock::now();
        auto t0 = Clock::now();
        std::vector<float> noise = generate_gaussian_noise(num_elements, sigma);
        auto t1 = Clock::now();
        tm.t_noise_gen_ms = ms_since(t0, t1);


        // 阶段 1: 正向扰动 (W + z)
        t0 = Clock::now();
        for (int i = 0; i < num_elements; ++i) {
            float val = fp16_to_fp32(base_weights[i]);
            val += noise[i]; 
            weight_ptr[i] = fp32_to_fp16(val);
        }
        t1 = Clock::now();
        tm.t_perturb_pos_ms = ms_since(t0, t1);

        t0 = Clock::now();
        ctx->memory->clear(true);
        t1 = Clock::now();
        tm.t_clear_pos_ms = ms_since(t0, t1);

        t0 = Clock::now();
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "inference failed (+)\n"); return 1; }
        t1 = Clock::now();
        tm.t_decode_pos_ms = ms_since(t0, t1);

        float loss_pos = -llama_get_logits_ith(ctx, n_tokens - 1)[target_label_token];

        // 阶段 2: 负向扰动 (W - z)
        t0 = Clock::now();
        for (int i = 0; i < num_elements; ++i) {
            float val = fp16_to_fp32(base_weights[i]);
            val -= noise[i]; // 相对 base 是 -z，相对当前(+z)是 -2z，这里重新基于 base 计算更稳妥
            weight_ptr[i] = fp32_to_fp16(val);
        }
        t1 = Clock::now();
        tm.t_perturb_neg_ms = ms_since(t0, t1);

        // 再次清除 KV Cache
        t0 = Clock::now();
        ctx->memory->clear(true);
        t1 = Clock::now();
        tm.t_clear_neg_ms = ms_since(t0, t1);

        t0 = Clock::now();
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "inference failed (-)\n"); return 1; }
        t1 = Clock::now();
        tm.t_decode_neg_ms = ms_since(t0, t1);

        float loss_neg = -llama_get_logits_ith(ctx, n_tokens - 1)[target_label_token];

        // 阶段 3: 更新
        float loss_diff = loss_pos - loss_neg;
        float grad_est = loss_diff / (2.0f * sigma);
        printf("Step %d | L+: %.4f | L-: %.4f | Diff: %.4f\n", step, loss_pos, loss_neg, loss_diff);

        t0 = Clock::now();
        for (int i = 0; i < num_elements; ++i) {
            float w = fp16_to_fp32(base_weights[i]);
            w -= learning_rate * grad_est * noise[i];
            base_weights[i] = fp32_to_fp16(w);
            weight_ptr[i] = base_weights[i];
        }
        t1 = Clock::now();
        tm.t_update_ms = ms_since(t0, t1);
        
        auto t_step1 = Clock::now();
        tm.t_step_total_ms = ms_since(t_step0, t_step1);

        tm.print(step);
    }

    // 清理
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
