import json

def minify_dataset(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cleaned_data = []
        for item in data:
            # 确保 output 是被严格压缩的 JSON 字符串（无空格，无换行）
            if isinstance(item['output'], dict):
                # separators=(',', ':') 会去掉所有多余空格
                minified_json = json.dumps(item['output'], ensure_ascii=False, separators=(',', ':'))
                item['output'] = minified_json
            elif isinstance(item['output'], str):
                # 如果已经是字符串，先解析再压缩
                try:
                    parsed = json.loads(item['output'])
                    item['output'] = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
                except:
                    pass
            cleaned_data.append(item)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2) # 整体可以用indent，但output字段内部已经是单行字符串了
            
        print(f"✅ 成功压缩数据: {input_file} -> {output_file}")
    except Exception as e:
        print(f"⚠️ 处理 {input_file} 时出错: {e}")

if __name__ == "__main__":
    # 请替换为你实际的训练集路径
    minify_dataset("data/processed/training/train_seq2seq_clean.json", "data/processed/training/train_seq2seq_minified.json")
    minify_dataset("data/processed/training/val_seq2seq_clean.json", "data/processed/training/val_seq2seq_minified.json")
    minify_dataset("data/processed/training/test_seq2seq_clean.json", "data/processed/training/test_seq2seq_minified.json")