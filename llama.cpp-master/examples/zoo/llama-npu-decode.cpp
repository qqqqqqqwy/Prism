#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

// 辅助函数：忽略大小写检查字符串包含关系
static bool contains_icase(std::string s, std::string p) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    std::transform(p.begin(), p.end(), p.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s.find(p) != std::string::npos;
}

// 识别 NPU 设备（如 Qualcomm Hexagon/HTP 或通用 QNN 后端）
static bool is_npu_device(ggml_backend_dev_t dev) {
    const char * n0 = ggml_backend_dev_name(dev);
    const char * n1 = ggml_backend_dev_description(dev);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    const char * n2 = buft ? ggml_backend_buft_name(buft) : "";
    std::string key = std::string(n0 ? n0 : "") + " " + std::string(n1 ? n1 : "") + " " + std::string(n2 ? n2 : "");
    return contains_icase(key, "hexagon") || contains_icase(key, "htp") || contains_icase(key, "npu") || contains_icase(key, "qnn");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];

    // 1. 初始化后端
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    ggml_backend_load_all();

    // 2. 寻找 NPU 设备
    ggml_backend_dev_t npu_dev = nullptr;
    std::printf("[NPU] Scanning available devices:\n");
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        std::printf("  - %s | %s\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        if (!npu_dev && is_npu_device(dev)) {
            npu_dev = dev;
        }
    }

    if (!npu_dev) {
        std::printf("[NPU] ERROR: No NPU device found. Please check your runtime environment.\n");
        llama_backend_free();
        return 2;
    }

    std::printf("[NPU] Using device: %s\n", ggml_backend_dev_name(npu_dev));

    // 3. 配置模型参数：将所有层卸载到 NPU
    ggml_backend_dev_t devices[2] = { npu_dev, nullptr };
    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = false;
    model_params.devices = devices;
    model_params.n_gpu_layers = -1; // 强制所有层到 NPU
    model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;

    struct llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        std::fprintf(stderr, "Error: failed to load model\n");
        return 1;
    }

    // 4. 初始化上下文
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    ctx_params.n_threads = 5;       // NPU 负载下 CPU 线程主要负责调度
    ctx_params.n_threads_batch = 5;

    struct llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::fprintf(stderr, "Error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // 5. 构造测试输入 (30 tokens)
    std::vector<llama_token> tokens;
    for (int i = 0; i < 30; ++i) {
        tokens.push_back(1 + i);
    }
    int n_tokens = (int)tokens.size();

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]  = tokens[i];
        batch.pos[i]    = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1);
    }

    printf("\n[INFO] Starting NPU Benchmark: Prefill vs Decoding...\n");

    // 6. Prefill 阶段
    auto t_pre_start = std::chrono::steady_clock::now();
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Prefill failed\n");
        return 1;
    }
    auto t_pre_end = std::chrono::steady_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_pre_end - t_pre_start).count();
    printf("[RESULT] Prefill (30 tokens) time: %.3f ms (%.3f ms/tok)\n", prefill_ms, prefill_ms / 30.0);

    // 7. Decoding 阶段 (逐 Token 生成)
    int n_decode = 20; 
    double total_decode_ms = 0.0;
    llama_token last_token = 1;

    for (int i = 0; i < n_decode; ++i) {
        llama_batch s_batch = llama_batch_init(1, 0, 1);
        s_batch.n_tokens = 1;
        s_batch.token[0]    = last_token;
        s_batch.pos[0]      = n_tokens + i;
        s_batch.n_seq_id[0] = 1;
        s_batch.seq_id[0][0] = 0;
        s_batch.logits[0]   = true;

        auto t_dec_start = std::chrono::steady_clock::now();
        
        // NPU 执行推理
        if (llama_decode(ctx, s_batch) != 0) {
            printf("Decode failed at step %d\n", i);
            llama_batch_free(s_batch);
            break;
        }
        
        auto t_dec_end = std::chrono::steady_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(t_dec_end - t_dec_start).count();
        
        total_decode_ms += step_ms;
        printf("  Token %d | NPU Decode time: %.3f ms\n", i + 1, step_ms);
        
        llama_batch_free(s_batch);
    }

    printf("\n[RESULT] Average Single Token Decode time (NPU): %.3f ms\n", total_decode_ms / n_decode);
    printf("[RESULT] Generation speed: %.2f tokens/s\n", 1000.0 / (total_decode_ms / n_decode));

    // 清理
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}