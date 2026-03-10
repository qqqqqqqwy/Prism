# Prism: Efficient Zeroth-Order Fine-Tuning for On-Device LLMs

Prism is a mobile-native infrastructure designed to enable efficient Zeroth-Order (ZO) fine-tuning for billion-scale Large Language Models (LLMs) directly on edge devices (e.g., smartphones). By re-architecting the ZO optimization loop to fit within strict mobile memory, latency, and power budgets, Prism makes on-device adaptation practical without the massive overhead of backpropagation.

## Key Innovations

Prism addresses the critical memory bottlenecks of on-device backpropagation and the slow convergence typically associated with ZO optimization in high-dimensional spaces. The system features several core algorithmic and system-level innovations:

- **Row-Column Noise Injection:** Optimizes the generation of perturbations in high-dimensional parameter spaces to ensure efficient and scalable probing.
- **Mixed Quantization Strategy:** Drastically compresses training states and memory footprint without sacrificing fine-tuning accuracy.
- **Stochastic Directional Screening (SDS):** An early-exit mechanism that significantly accelerates the evaluation and update iteration of ZO gradients.

## Repository Structure

This repository is organized into two main components, separating the server-side experimental simulations from the actual on-device deployment:

- **`server/`**: The Python-based training framework and benchmarking suite. It contains scripts for evaluating various architectures (e.g., LLaMA, Phi, Qwen, TinyLlama), comparing First-Order (FO) and ZO optimization. For setup and execution details, refer to [`server/README.md`](./server/README.md).
- **`llama.cpp-master/`**: The mobile-native C++ inference and fine-tuning engine, built upon `llama.cpp`. This directory includes the core ZO operators, in-place memory management, NPU backend acceleration, and energy profiling integrations. For cross-compilation guidelines (Android NDK) and on-device execution, refer to [`llama.cpp-master/README.md`](./llama.cpp-master/README.md).

## Hardware Evaluation

Prism has been extensively evaluated and benchmarked on commodity Android smartphones, demonstrating exceptionally low energy consumption and peak memory optimization on the following flagship devices:
- Redmi K60 Pro
- Redmi K70 Ultra
- OnePlus 12
- OnePlus Ace 3

## Getting Started

Because the server-side simulation (Python) and the on-device engine (C/C++) require entirely different toolchains and dependencies, please consult the specific documentation in each sub-directory:

1. **Server-Side Evaluation:** Navigate to the `server/` directory for data loading, environment setup, and batch experiment scripts.
2. **On-Device Deployment:** Navigate to the `llama.cpp-master/` directory for instructions on cross-compiling the engine and launching fine-tuning tasks via the Android shell.
