#!/usr/bin/env python3
"""
预下载模型到本地缓存，避免训练时网络超时
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "t5-small"  # 或你需要的模型

print(f"正在下载模型: {MODEL_NAME}")
print("使用镜像源: https://hf-mirror.com")

try:
    print("1/2 下载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("2/2 下载 model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    print("✅ 模型下载完成！")
    print(f"缓存位置: ~/.cache/huggingface/")
    print("现在可以离线训练了")
    
except Exception as e:
    print(f"❌ 下载失败: {e}")
    print("尝试其他镜像源...")
    
    # 备用源
    mirrors = [
        "https://hf-mirror.com",
        "https://huggingface.co",
    ]
    
    for mirror in mirrors:
        print(f"\n尝试: {mirror}")
        os.environ['HF_ENDPOINT'] = mirror
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            print("✅ 成功！")
            break
        except Exception as e2:
            print(f"失败: {e2}")