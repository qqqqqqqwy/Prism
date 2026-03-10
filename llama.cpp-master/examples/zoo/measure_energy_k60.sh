#!/bin/bash

TESTS=(
    # "uv-tinyllama;./llama-zoo-uv-50step tinyllama-1.1b-chat.gguf"
    # "uv-qwen;./llama-zoo-uv-50step qwen-2.5-3b-f16.gguf"
    # "uv-phi;./llama-zoo-uv-50step phi-2-3b-f16.gguf"
    # "uv-meta;./llama-zoo-uv-50step llama-3.2-3b-f16.gguf"
    "all-tinyllama;./llama-zoo-all-50step tinyllama-1.1b-chat.gguf"
    "all-qwen;./llama-zoo-all-50step qwen-2.5-3b-f16.gguf"
    "all-phi;./llama-zoo-all-50step phi-2-3b-f16.gguf"
    "all-llama;./llama-zoo-all-50step llama-3.2-3b-f16.gguf"
)

TOTAL_TESTS=${#TESTS[@]}

for i in "${!TESTS[@]}"; do
    IFS=";" read -r PREFIX CMD <<< "${TESTS[$i]}"
    
    echo "------------------------------------------------"
    echo "[$((i+1))/$TOTAL_TESTS] 开始测试: $PREFIX"
    
    # 记录前快照
    adb -s 192.168.137.114 shell dumpsys batterystats > "${PREFIX}-before.txt"
    
    # 执行程序
    echo " -> 正在运行 $CMD ..."
    adb -s 192.168.137.114 shell "cd /data/local/tmp/llama-zoo && $CMD"
    
    # 记录后快照
    adb -s 192.168.137.114 shell dumpsys batterystats > "${PREFIX}-after.txt"
    
    echo " -> $PREFIX 运行结束."

    # 如果不是最后一个测试，则等待 120 秒冷却
    if [ $i -lt $((TOTAL_TESTS - 1)) ]; then
        echo " -> 冷却 120 秒..."
        sleep 120
    fi
done

# echo "------------------------------------------------"
# echo "=== 恢复设备充电状态 ==="
# adb shell dumpsys battery reset

echo ""
echo "================ 最终能耗测试结果 ================"

for TEST in "${TESTS[@]}"; do
    IFS=";" read -r PREFIX CMD <<< "$TEST"
    BEFORE_FILE="${PREFIX}-before.txt"
    AFTER_FILE="${PREFIX}-after.txt"
    
    # 使用 grep 提取带有 "Uid 2000:" 的行（-m 1 确保只取第一次出现，-i 忽略大小写）
    # 使用 awk '{print $3}' 提取第三个字段 (即具体的数值)
    BEFORE_VAL=$(grep -m 1 -i "uid 2000:" "$BEFORE_FILE" | awk '{print $3}')
    AFTER_VAL=$(grep -m 1 -i "uid 2000:" "$AFTER_FILE" | awk '{print $3}')

    # 如果提取为空 (比如耗电极小被系统忽略)，则默认设为 0
    if [ -z "$BEFORE_VAL" ]; then BEFORE_VAL=0; fi
    if [ -z "$AFTER_VAL" ]; then AFTER_VAL=0; fi

    # 使用 awk 计算浮点数差值
    DIFF=$(awk -v a="$AFTER_VAL" -v b="$BEFORE_VAL" 'BEGIN { printf "%.3f", a - b }')

    # 输出到终端
    echo "${PREFIX}: ${DIFF} mAh"
done
echo "=================================================="