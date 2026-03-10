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

const std::vector<std::pair<std::string, struct ggml_tensor *>> & llama_internal_get_tensor_map(const struct llama_model * model);

static bool contains_icase(std::string s, std::string p) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    std::transform(p.begin(), p.end(), p.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s.find(p) != std::string::npos;
}

static bool is_npu_device(ggml_backend_dev_t dev) {
    const char * n0 = ggml_backend_dev_name(dev);
    const char * n1 = ggml_backend_dev_description(dev);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    const char * n2 = buft ? ggml_backend_buft_name(buft) : "";
    std::string key = std::string(n0 ? n0 : "") + " " + std::string(n1 ? n1 : "") + " " + std::string(n2 ? n2 : "");
    return contains_icase(key, "hexagon") || contains_icase(key, "htp") || contains_icase(key, "npu") || contains_icase(key, "qnn");
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <model.gguf> <1|2>\n", argv[0]);
        std::fprintf(stderr, "  1 = int4 model, 2 = int8 model\n");
        return 1;
    }

    const char * model_path = argv[1];
    const int quant_mode = std::atoi(argv[2]);
    if (quant_mode != 1 && quant_mode != 2) {
        std::fprintf(stderr, "Error: quant mode must be 1 (int4) or 2 (int8)\n");
        return 1;
    }

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    ggml_backend_load_all();

    std::printf("[NPU] Scanning available devices:\n");
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
        std::printf("  - %s | %s | buft=%s\n",
            ggml_backend_dev_name(dev),
            ggml_backend_dev_description(dev),
            buft ? ggml_backend_buft_name(buft) : "(null)");
    }

    ggml_backend_dev_t npu_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (is_npu_device(dev)) {
            npu_dev = dev;
            break;
        }
    }

    if (!npu_dev) {
        std::printf("[NPU] NOT USED: no NPU/Hexagon backend device found.\n");
        llama_backend_free();
        return 2;
    }

    std::printf("[NPU] Selected device: %s (%s)\n",
        ggml_backend_dev_name(npu_dev),
        ggml_backend_dev_description(npu_dev));

    ggml_backend_dev_t devices[2] = { npu_dev, nullptr };

    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = false;
    model_params.devices = devices;
    model_params.n_gpu_layers = -1; // offload all layers to selected device(s)
    model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
    model_params.main_gpu = 0;

    std::printf("[RUN] Loading model: %s (quant_mode=%d)\n", model_path, quant_mode);
    struct llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        std::fprintf(stderr, "Error: failed to load model '%s'\n", model_path);
        llama_backend_free();
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    ctx_params.n_threads = 5;
    ctx_params.n_threads_batch = 5;

    struct llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::fprintf(stderr, "Error: failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    size_t npu_tensor_count = 0;
    size_t npu_tensor_bytes = 0;
    const auto & tensors = llama_internal_get_tensor_map(model);
    for (const auto & kv : tensors) {
        struct ggml_tensor * t = kv.second;
        if (!t || !t->buffer) {
            continue;
        }
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev == npu_dev) {
            ++npu_tensor_count;
            npu_tensor_bytes += ggml_nbytes(t);
        }
    }

    if (npu_tensor_count > 0) {
        std::printf("[NPU] USED: tensors on NPU device = %zu (%.2f MiB)\n",
            npu_tensor_count, npu_tensor_bytes / 1024.0 / 1024.0);
    } else {
        std::printf("[NPU] NOT USED: no model tensor allocated on selected NPU device.\n");
    }
    llama_memory_breakdown_print(ctx);

    std::vector<llama_token> tokens;
    tokens.reserve(30);
    for (int i = 0; i < 30; ++i) {
        tokens.push_back(1 + i);
    }

    const int n_tokens = (int) tokens.size();
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1);
    }

    llama_memory_t mem = llama_get_memory(ctx);

    // warmup
    llama_memory_clear(mem, true);
    const int warmup_ret = llama_decode(ctx, batch);
    if (warmup_ret != 0 && warmup_ret != 1) {
        std::fprintf(stderr, "Error: warmup decode failed, ret=%d\n", warmup_ret);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    std::printf("[RUN] Warmup done.\n");

    using clock_t = std::chrono::steady_clock;
    std::vector<double> times_ms;
    times_ms.reserve(10);

    for (int i = 0; i < 10; ++i) {
        llama_memory_clear(mem, true);
        const auto t0 = clock_t::now();
        const int ret = llama_decode(ctx, batch);
        const auto t1 = clock_t::now();
        if (ret != 0 && ret != 1) {
            std::fprintf(stderr, "Error: decode failed at iter %d, ret=%d\n", i, ret);
            break;
        }

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(ms);
        std::printf("[RUN] iter %02d: %.3f ms\n", i + 1, ms);
    }

    double avg_ms = 0.0;
    for (double t : times_ms) {
        avg_ms += t;
    }
    if (!times_ms.empty()) {
        avg_ms /= (double) times_ms.size();
    }

    std::printf("[RUN] average over %zu runs: %.3f ms\n", times_ms.size(), avg_ms);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return times_ms.size() == 10 ? 0 : 3;
}
