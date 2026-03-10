adb -s b49f281b shell "cd /data/local/tmp && rm -rf llama-zoo && mkdir llama-zoo"
adb -s b49f281b push ./* /data/local/tmp/llama-zoo
adb -s b49f281b shell "cd /data/local/tmp/llama-zoo && chmod +x llama-zoo-all && chmod +x llama-zoo-uv"
adb -s b49f281b shell "cd /data/local/tmp/llama-zoo && chmod +x llama-zoo-uv-50step && chmod +x llama-zoo-all-50step"