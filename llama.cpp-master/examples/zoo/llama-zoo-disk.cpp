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
            "      noise_gen:   %.3f ms\n"
            "      +perturb:    %.3f ms | +clear: %.3f ms | +decode: %.3f ms\n"
            "      -perturb:    %.3f ms | -clear: %.3f ms | -decode: %.3f ms\n"
            "      update:      %.3f ms\n",
            step, t_step_total_ms,
            t_noise_gen_ms,
            t_perturb_pos_ms, t_clear_pos_ms, t_decode_pos_ms,
            t_perturb_neg_ms, t_clear_neg_ms, t_decode_neg_ms,
            t_update_ms
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

void generate_and_save_noise_to_disk(const std::vector<LayerInfo>& layers, float std_dev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0.0f, std_dev);

    // 找到最大的层大小，分配唯一的 Buffer
    int64_t max_size = 0;
    for (const auto& l : layers) if (l.count > max_size) max_size = l.count;
    std::vector<float> noise_buffer(max_size);

    for (const auto& layer : layers) {
        for (int64_t i = 0; i < layer.count; ++i) {
            noise_buffer[i] = d(gen);
        }
        std::string path = get_noise_path(layer.name);
        std::ofstream outfile(path, std::ios::binary);
        if (outfile) {
            outfile.write(reinterpret_cast<const char*>(noise_buffer.data()), layer.count * sizeof(float));
            outfile.close();
        }
    }
}

void apply_noise_scale(std::vector<LayerInfo>& layers, float scale) {
    if (scale == 0.0f) return;

    int64_t max_size = 0;
    for (const auto& l : layers) if (l.count > max_size) max_size = l.count;
    std::vector<float> noise_buffer(max_size);

    for (auto& layer : layers) {
        std::string path = get_noise_path(layer.name);
        std::ifstream infile(path, std::ios::binary);
        if (!infile) continue;
        
        infile.read(reinterpret_cast<char*>(noise_buffer.data()), layer.count * sizeof(float));
        infile.close();

        ggml_fp16_t* current_ptr = (ggml_fp16_t*)layer.tensor->data;

        for (int64_t i = 0; i < layer.count; ++i) {
            float val = fp16_to_fp32(current_ptr[i]);
            val += scale * noise_buffer[i];
            current_ptr[i] = fp32_to_fp16(val);
        }
    }
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
        generate_and_save_noise_to_disk(target_layers, sigma);
        t_temp_end = Clock::now();
        tm.t_noise_gen_ms = ms_since(t_temp_start, t_temp_end);

        t_temp_start = Clock::now();
        apply_noise_scale(target_layers, 1.0f);
        t_temp_end = Clock::now();
        tm.t_perturb_pos_ms = ms_since(t_temp_start, t_temp_end);

        t_temp_start = Clock::now();
        ctx->memory->clear(true);
        t_temp_end = Clock::now();
        tm.t_clear_pos_ms = ms_since(t_temp_start, t_temp_end);

        t_temp_start = Clock::now();
        if (llama_decode(ctx, batch) != 0) break;
        t_temp_end = Clock::now();
        tm.t_decode_pos_ms = ms_since(t_temp_start, t_temp_end);
        
        float loss_pos = -llama_get_logits_ith(ctx, n_tokens - 1)[target_label_token];

        t_temp_start = Clock::now();
        apply_noise_scale(target_layers, -2.0f);
        t_temp_end = Clock::now();
        tm.t_perturb_neg_ms = ms_since(t_temp_start, t_temp_end);

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
        
        t_temp_start = Clock::now();
        float restore_and_update_scale = 1.0f - (learning_rate * grad_est);
        apply_noise_scale(target_layers, restore_and_update_scale);
        t_temp_end = Clock::now();
        tm.t_update_ms = ms_since(t_temp_start, t_temp_end);

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
