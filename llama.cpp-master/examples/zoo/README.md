# Prism On-Device Engine

This directory contains the core on-device fine-tuning engine for the Prism project. The complete Zeroth-Order (ZO) fine-tuning logic is implemented in `examples/zoo/llama-zoo.cpp`. 

## Environment Setup

The following compilation and execution guidelines are based on a **Windows 11** PC environment with **WSL (Windows Subsystem for Linux)** installed. The on-device deployment consists of three main stages: Executable Compilation, Model Preparation, and File Transfer & Execution.

## 1. Executable Compilation

Since the engine runs natively on Android devices, cross-compilation is required using the Android NDK.

1. **Download the NDK:** Download `android-ndk-r26c` to your local WSL environment.
2. **Configure the Environment Variable:** Open the compilation script (`build.sh`) and configure the NDK path before compiling:
   ```bash
   export ANDROID_NDK_ROOT=[Your Own Path]/android-ndk-r26c
   ```
3. **Compile the Executable:** Navigate to the project directory and run the build script:
   ```bash
   cd llama.cpp-master/examples/zoo
   sh build.sh
   ```

## 2. Model Preparation

After compilation, you need to prepare the dataset and convert the pre-trained model (using TinyLlama as an example) into the format required by the engine.

```bash
# Navigate to the llama.cpp-master root directory
cd ../..

# Create and activate a Python virtual environment
python3 -m venv llama
source ./llama/bin/activate

# Install necessary dependencies
pip install gguf
pip install -U "huggingface_hub[cli]"
pip install -U huggingface_hub
pip install datasets

# Download the pre-trained TinyLlama model to a local directory
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local_dir [Your Model Path] 

# Convert the Hugging Face model to GGUF format with f16 precision
python convert_hf_to_gguf.py [Your Model Path] --outfile examples/zoo/tinyllama-1.1b-chat.gguf --outtype f16

# Navigate back to the zoo directory and export the SST-2 dataset
cd examples/zoo
python export_sst2_tsv.py
```

## 3. File Transfer and Execution

For the actual on-device execution, connect your Android smartphone to the PC via USB and enable **USB Debugging**. Then, run the following commands in your PC terminal to deploy and launch the fine-tuning task:

```bash
# Create an independent working directory on the device via adb
adb shell "cd /data/local/tmp && rm -rf llama-zoo && mkdir llama-zoo"

# Push the converted model and necessary shared libraries to the device
adb push ./tinyllama-1.1b-chat.gguf /data/local/tmp/llama-zoo
adb push ./libc++_shared.so /data/local/tmp/llama-zoo
adb push ./libomp.so /data/local/tmp/llama-zoo
adb push ./sst2_eval.tsv /data/local/tmp/llama-zoo
adb push ./sst2_train.tsv /data/local/tmp/llama-zoo

# Navigate to the build directory and push the compiled executable
cd build
adb push ./llama-zoo /data/local/tmp/llama-zoo

# Enter the device shell
adb shell

# Grant execution permissions and configure the library path on the device
cd /data/local/tmp/llama-zoo
chmod +x llama-zoo
export LD_LIBRARY_PATH=/data/local/tmp/llama-zoo

# Launch the on-device fine-tuning process
./llama-zoo tinyllama-1.1b-chat.gguf sst2_train.tsv sst2_eval.tsv \
   --batch_size 4 --n_steps 5000 --lr 1e-7 --sigma 0.001 \
   --max_length 128 --eval_interval 500 --n_threads 5
```
