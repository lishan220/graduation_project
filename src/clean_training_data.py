import json
import re

def is_english(text):
    """宽松判断：允许英文字母、数字、常见标点及JSON特殊字符"""
    # 允许的字符集：字母、数字、空格、常见英文标点、JSON符号
    allowed = re.compile(r'^[a-zA-Z0-9\s.,;:!?\-_"\'\{\}\(\)\[\]/\\]+$')
    return bool(allowed.match(text))

def clean_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned = []
    discarded = 0
    for idx, item in enumerate(data):
        output = item['output']
        if is_english(output):
            cleaned.append(item)
        else:
            discarded += 1
            if discarded <= 5:  # 只打印前5个被丢弃的样本
                print(f"\n被丢弃的样本 #{discarded} (索引 {idx}):")
                print(f"output 预览: {repr(output[:200])}")  # 使用 repr 显示特殊字符
    print(f"\n原始样本数: {len(data)}")
    print(f"清洗后样本数: {len(cleaned)}")
    print(f"丢弃样本数: {discarded}")

if __name__ == "__main__":
    clean_dataset("data/processed/training/train_seq2seq.json", "data/processed/training/train_seq2seq_clean.json")
    clean_dataset("data/processed/training/val_seq2seq.json", "data/processed/training/val_seq2seq_clean.json")
    clean_dataset("data/processed/training/test_seq2seq.json", "data/processed/training/test_seq2seq_clean.json")