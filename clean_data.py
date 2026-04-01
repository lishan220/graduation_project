#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清洗训练数据：将中文词替换为英文，避免模型学习到错误语言模式
"""

import json
import os

def clean_text(text):
    """替换常见中文词为英文"""
    replacements = {
        "用例": "case",
        "正常值": "normal",
        "边界值": "boundary",
        "异常值": "exception",
        "必填": "required",
        "可选": "optional",
        "字符串": "string",
        "整数": "integer",
        # 可根据需要添加更多替换
    }
    for zh, en in replacements.items():
        text = text.replace(zh, en)
    return text

def clean_dataset(input_file, output_file):
    print(f"处理 {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned = []
    for item in data:
        # 清洗 input 和 output 字段
        item['input'] = clean_text(item['input'])
        item['output'] = clean_text(item['output'])
        cleaned.append(item)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"  原始样本数: {len(data)}")
    print(f"  清洗后样本数: {len(cleaned)}")

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("data/processed/training", exist_ok=True)

    clean_dataset("data/processed/training/train_seq2seq.json", "data/processed/training/train_seq2seq_clean.json")
    clean_dataset("data/processed/training/val_seq2seq.json", "data/processed/training/val_seq2seq_clean.json")
    clean_dataset("data/processed/training/test_seq2seq.json", "data/processed/training/test_seq2seq_clean.json")

    print("\n清洗完成！生成的清洗文件：")
    print("  - data/processed/training/train_seq2seq_clean.json")
    print("  - data/processed/training/val_seq2seq_clean.json")
    print("  - data/processed/training/test_seq2seq_clean.json")