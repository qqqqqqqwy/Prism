#!/bin/bash

TESTS=(
    "uv-tinyllama;./llama-zoo-uv tinyllama-1.1b-chat.gguf"
    "uv-qwen;./llama-zoo-uv qwen-2.5-3b-f16.gguf"
    "uv-phi;./llama-zoo-uv phi-2-3b-f16.gguf"
    "uv-meta;./llama-zoo-uv llama-3.2-3b-f16.gguf"
    "all-tinyllama;./llama-zoo-all tinyllama-1.1b-chat.gguf"
    "all-qwen;./llama-zoo-all qwen-2.5-3b-f16.gguf"
    "all-phi;./llama-zoo-all phi-2-3b-f16.gguf"
    "all-llama;./llama-zoo-all llama-3.2-3b-f16.gguf"
)
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/local/tmp/llama-zoo

TOTAL_TESTS=${#TESTS[@]}

for i in "${!TESTS[@]}"; do
    IFS=";" read -r PREFIX CMD <<< "${TESTS[$i]}"

    echo "=================================================="
    echo "[$((i+1))/$TOTAL_TESTS] 开始执行并实时监控..."
    echo "测试项: $PREFIX"
    echo "命令: $CMD"
    echo "=================================================="

    $CMD &
    PID=$!

    PEAK_MEM_KB=0
    PEAK_POWER_W=0

    while [ -d /proc/$PID ]; do
        CUR_PEAK_MEM=$(cat /proc/$PID/status 2>/dev/null | grep VmHWM | awk '{print $2}')
        if [ -n "$CUR_PEAK_MEM" ] && [ "$CUR_PEAK_MEM" -gt "$PEAK_MEM_KB" ]; then
            PEAK_MEM_KB=$CUR_PEAK_MEM
        fi

        # 采样间隔 50ms
        sleep 0.05
    done

    wait $PID
    EXIT_CODE=$?

    PEAK_MEM_MB=$(awk "BEGIN {printf \"%.2f\", $PEAK_MEM_KB / 1024}")

    echo ""
    echo "=================================================="
    echo "性能评估报告 - $PREFIX"
    echo "=================================================="
    if [ $EXIT_CODE -ne 0 ]; then
        echo "程序异常退出，状态码 $EXIT_CODE"
    fi
    echo "峰值物理内存占用 (Peak RSS): $PEAK_MEM_MB MB"
    echo "=================================================="

    # 如果不是最后一个测试，可选冷却（如不需要可删除）
    if [ $i -lt $((TOTAL_TESTS - 1)) ]; then
        echo "冷却 5 秒..."
        sleep 5
    fi
done