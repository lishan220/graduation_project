import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

MODEL_PATH = "models/fine_tuned_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    print(f"加载模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE)
    model.eval()
    return tokenizer, model

def generate(desc, tokenizer, model, max_len=128):
    inputs = tokenizer(desc, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_len,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    tokenizer, model = load_model()
    
    test_descriptions = [
        "生成测试用例JSON: POST /users - 创建用户，需要username（字符串，必填，3-20字符）和age（整数，可选，0-120）",
        "生成测试用例JSON: GET /pet/{petId} - 通过ID获取宠物信息，petId为整数",
        "生成测试用例JSON: DELETE /users/{id} - 根据ID删除用户",
    ]
    
    for desc in test_descriptions:
        print("\n" + "="*60)
        print(f"输入: {desc}")
        output = generate(desc, tokenizer, model)
        print(f"生成: {output}")
        
        # 尝试解析JSON
        try:
            parsed = json.loads(output)
            print("✅ 有效JSON:")
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"❌ 不是有效JSON，错误: {e}")
            print("原始输出:", output)
            
            # 尝试提取可能的大括号内容
            import re
            match = re.search(r'\{.*\}', output, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                    print("✅ 提取后有效JSON:")
                    print(json.dumps(parsed, indent=2, ensure_ascii=False))
                except:
                    print("提取后仍然不是有效JSON")

if __name__ == "__main__":
    main()