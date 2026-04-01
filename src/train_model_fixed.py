#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5模型微调脚本 - 继续训练版 (针对 T5-small 优化)
增加目标长度，防止结构化数据被提前截断
"""

import os
import sys
import json
import torch
import random
import numpy as np

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict

# ========== 核心配置 ==========
MODEL_NAME = "t5-small"
OUTPUT_DIR = "models/fine_tuned_model"
DATA_DIR = "data/processed/training"
LOG_DIR = "logs"

MAX_INPUT_LENGTH = 256
# 💡 核心修改 1：进一步提升目标长度到 1024，给 T5 足够的空间生成完整 JSON
MAX_TARGET_LENGTH = 1024  
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS = 5                       
LEARNING_RATE = 3e-5             
WARMUP_STEPS = 50
SAVE_STEPS = 1000
EVAL_STEPS = 200
LOGGING_STEPS = 20
PATIENCE = 100                    
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(RANDOM_SEED)

def load_datasets():
    data_files = {
        "train": os.path.join(DATA_DIR, "train_seq2seq_minified.json"),
        "validation": os.path.join(DATA_DIR, "val_seq2seq_minified.json"),
        "test": os.path.join(DATA_DIR, "test_seq2seq_minified.json")
    }
    datasets = {}
    for split, file_path in data_files.items():
        if not os.path.exists(file_path):
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                dataset = Dataset.from_list(data)
            elif isinstance(data, dict):
                dataset = Dataset.from_dict(data)
            else:
                continue
            datasets[split] = dataset
        except Exception as e:
            print(f"加载失败: {e}")
            
    if "validation" not in datasets and "train" in datasets:
        train_val = datasets["train"].train_test_split(test_size=0.2, seed=RANDOM_SEED)
        datasets["train"] = train_val["train"]
        datasets["validation"] = train_val["test"]
    return DatasetDict(datasets)

def preprocess_function(examples, tokenizer):
    model_inputs = tokenizer(examples['input'], max_length=MAX_INPUT_LENGTH, truncation=True)
    labels = tokenizer(examples['output'], max_length=MAX_TARGET_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    print("=" * 60)
    print(f"启动 T5 ({MODEL_NAME}) 微调 - 增强 JSON 生成能力")
    print("=" * 60)

    checkpoint = None
    if os.path.exists(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith('checkpoint-')]
        if checkpoints:
            latest = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
            checkpoint = os.path.join(OUTPUT_DIR, latest)
            print(f"🔄 找到检查点: {latest}，将继续训练")

    raw_datasets = load_datasets()

    print("\n🔧 加载模型与分词器...")
    try:
        if os.path.exists(os.path.join(OUTPUT_DIR, "config.json")):
            tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR, local_files_only=True)
            checkpoint = None 
        else:
            raise Exception("未找到本地已微调模型")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    model = model.to(DEVICE)

    tokenized_datasets = raw_datasets.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        logging_dir=LOG_DIR,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps" if "validation" in tokenized_datasets else "no",
        eval_steps=EVAL_STEPS,
        load_best_model_at_end=True if "validation" in tokenized_datasets else False,
        fp16=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)] if "validation" in tokenized_datasets else [],
    )

    print("\n🏃 开始训练...")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("\n✅ 训练与保存完成!")

if __name__ == "__main__":
    main()