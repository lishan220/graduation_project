#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5模型微调脚本 - 继续训练版
从上次检查点继续，禁用早停，自动恢复
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

# ========== 配置 ==========
MODEL_NAME = "t5-small"
OUTPUT_DIR = "models/fine_tuned_model"
DATA_DIR = "data/processed/training"
LOG_DIR = "logs"

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 512
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS = 15                      # 总epoch数（如果已训练9，则再跑6）
LEARNING_RATE = 3e-5             # 降低学习率，更稳定
WARMUP_STEPS = 50
SAVE_STEPS = 1000
EVAL_STEPS = 200
LOGGING_STEPS = 20
PATIENCE = 100                    # 禁用早停（设很大）
RANDOM_SEED = 42

DEVICE = torch.device("cpu")
torch.set_num_threads(4)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(RANDOM_SEED)

def load_datasets():
    data_files = {
        "train": os.path.join(DATA_DIR, "train_seq2seq_clean.json"),
        "validation": os.path.join(DATA_DIR, "val_seq2seq_clean.json"),
        "test": os.path.join(DATA_DIR, "test_seq2seq_clean.json")
    }
    datasets = {}
    for split, file_path in data_files.items():
        if not os.path.exists(file_path):
            print(f"⚠️  {file_path} 不存在，跳过 {split}")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"\n📊 {split}: {len(data)} 个样本")
            if isinstance(data, list):
                dataset = Dataset.from_list(data)
            elif isinstance(data, dict):
                dataset = Dataset.from_dict(data)
            else:
                raise ValueError(f"不支持的数据格式: {type(data)}")
            if 'input' not in dataset.column_names or 'output' not in dataset.column_names:
                print(f"   ❌ 缺少 input/output 字段")
                continue
            datasets[split] = dataset
            print(f"   ✅ 加载成功")
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
            continue
    if not datasets:
        print("❌ 没有加载到任何数据！")
        sys.exit(1)
    if "validation" not in datasets and "train" in datasets:
        print("\n⚠️  从训练集划分20%作为验证集")
        train_val = datasets["train"].train_test_split(test_size=0.2, seed=RANDOM_SEED)
        datasets["train"] = train_val["train"]
        datasets["validation"] = train_val["test"]
    return DatasetDict(datasets)

def preprocess_function(examples, tokenizer):
    inputs = examples.get('input', [])
    targets = examples.get('output', [])
    if isinstance(inputs, str):
        inputs = [inputs]
    if isinstance(targets, str):
        targets = [targets]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False
    )
    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    print("=" * 60)
    print("T5模型微调 - 继续训练版")
    print("=" * 60)

    # 检查是否有检查点
    checkpoint = None
    if os.path.exists(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith('checkpoint-')]
        if checkpoints:
            # 找到最新的检查点（按步数排序）
            latest = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
            checkpoint = os.path.join(OUTPUT_DIR, latest)
            print(f"🔄 找到检查点: {latest}")
            # 粗略估算已训练步数：假设每epoch约1167步（根据之前经验）
            # 这一步只是提示，不影响实际恢复
            steps_done = int(latest.split('-')[1])
            epoch_done = steps_done / (len(load_datasets()["train"]) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
            print(f"   已训练约 {epoch_done:.2f} 个epoch，将从该点继续训练到 {EPOCHS} 个epoch")
        else:
            print("⚠️  没有找到检查点，将从头训练")
    else:
        print("⚠️  输出目录不存在，将从头训练")

    print(f"💻 CPU模式（线程数: {torch.get_num_threads()}）")
    # 预估剩余时间（仅当有检查点时粗略估算）
    if checkpoint:
        remaining_epochs = max(0, EPOCHS - epoch_done)
        print(f"⚠️  预计还需: {remaining_epochs:.2f} epochs × ~25分钟 = {remaining_epochs * 25:.0f}分钟")
    else:
        print(f"⚠️  预计总时间: {EPOCHS} epochs × ~25分钟 = {EPOCHS * 25}分钟")

    print("\n📂 加载数据...")
    raw_datasets = load_datasets()
    for split in raw_datasets:
        print(f"   {split}: {len(raw_datasets[split])} 样本")

    print(f"\n🔧 加载模型: {MODEL_NAME}")
    try:
        print("   尝试本地缓存...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, local_files_only=True)
        print("   ✅ 本地加载成功")
    except Exception:
        print("   尝试下载...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        print("   ✅ 下载成功")

    model = model.to(DEVICE)
    print(f"   参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    print("\n⚙️  预处理...")
    if "train" not in raw_datasets:
        print("❌ 训练集不存在")
        sys.exit(1)

    tokenized_datasets = raw_datasets.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    print("\n🎯 配置训练...")
    has_validation = "validation" in tokenized_datasets

    training_args_dict = {
        "output_dir": OUTPUT_DIR,
        "save_strategy": "steps",
        "save_steps": SAVE_STEPS,
        "save_total_limit": 2,          # 保留2个检查点，以防后续需要回退
        "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "num_train_epochs": EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "weight_decay": 0.01,
        "predict_with_generate": True,
        "generation_max_length": MAX_TARGET_LENGTH,
        "logging_dir": LOG_DIR,
        "logging_steps": LOGGING_STEPS,
        "fp16": False,
        "dataloader_num_workers": 0,
    }

    if has_validation:
        training_args_dict["eval_strategy"] = "steps"
        training_args_dict["eval_steps"] = EVAL_STEPS
        training_args_dict["load_best_model_at_end"] = True
        training_args_dict["metric_for_best_model"] = "eval_loss"
        training_args_dict["greater_is_better"] = False
    else:
        training_args_dict["eval_strategy"] = "no"

    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    callbacks = []
    if has_validation:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=PATIENCE))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print(f"\n🏃 开始训练（Epochs={EPOCHS}, 有效批次={BATCH_SIZE*GRADIENT_ACCUMULATION_STEPS}）")
    print("   按 Ctrl+C 可中断，模型会保存")
    print("-" * 60)

    try:
        # 关键修复：正确传递检查点路径
        result = trainer.train(resume_from_checkpoint=checkpoint)
        print(f"\n✅ 训练完成! Loss: {result.training_loss:.4f}")
    except KeyboardInterrupt:
        print("\n⚠️  中断，保存模型...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"✅ 已保存到 {OUTPUT_DIR}")
        sys.exit(130)

    print(f"\n💾 保存到 {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 测试生成
    print("\n📝 测试生成:")
    test_input = raw_datasets["train"][0]["input"]
    inputs = tokenizer(test_input, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=MAX_TARGET_LENGTH, num_beams=2)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   输入: {test_input[:60]}...")
    print(f"   生成: {generated[:60]}...")

    print("\n" + "=" * 60)
    print("🎉 完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()