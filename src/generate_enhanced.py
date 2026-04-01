import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "models/fine_tuned_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    print(f"加载模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE)
    model.eval()
    return tokenizer, model

def generate(desc, tokenizer, model, max_len=256):
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

def repair_json(raw):
    """
    尝试修复模型生成的近似JSON字符串
    """
    # 去除首尾空白
    raw = raw.strip()
    
    # 如果整个字符串不以 { 开头，尝试添加
    if not raw.startswith('{'):
        # 如果以 " 开头，可能整个就是一段键值对，加上外层花括号
        if raw.startswith('"'):
            raw = '{' + raw + '}'
        else:
            # 否则尝试寻找最外层的大括号
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                raw = match.group()
            else:
                raw = '{' + raw + '}'
    else:
        # 确保有闭合的大括号
        if not raw.endswith('}'):
            raw = raw + '}'
    
    # 替换不规范的分隔符 = 为 : （但要注意避免替换url中的等号？简单处理）
    # 只替换键值对之间的等号
    raw = re.sub(r'(\w+)\s*=\s*', r'\1: ', raw)
    
    # 修复引号不匹配的问题（简单尝试）
    # 确保键被双引号包围
    raw = re.sub(r'(\b\w+\b)(?=\s*:)', r'"\1"', raw)
    
    # 去除多余的逗号
    raw = re.sub(r',\s*}', '}', raw)
    raw = re.sub(r',\s*]', ']', raw)
    
    return raw

def main():
    tokenizer, model = load_model()
    
    test_descriptions = [
        "生成测试用例JSON: POST /users - 创建用户，需要username（字符串，必填，3-20字符）和age（整数，可选，0-120）",
        "生成测试用例JSON: GET /pet/{petId} - 通过ID获取宠物信息，petId为整数",
        "生成测试用例JSON: DELETE /users/{id} - 根据ID删除用户",
    ]
    
    for desc in test_descriptions:
        print("\n" + "="*70)
        print(f"输入: {desc}")
        output = generate(desc, tokenizer, model)
        print(f"原始生成: {output}")
        
        # 尝试修复
        fixed = repair_json(output)
        print(f"修复后: {fixed}")
        
        try:
            parsed = json.loads(fixed)
            print("✅ 成功解析为JSON:")
            print(json.dumps(parsed, indent=2, ensure_ascii=False)[:500])
        except json.JSONDecodeError as e:
            print(f"❌ 修复后仍无法解析: {e}")
            # 尝试提取可能的JSON片段
            match = re.search(r'\{.*\}', fixed, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                    print("✅ 提取后有效JSON:")
                    print(json.dumps(parsed, indent=2, ensure_ascii=False)[:500])
                except:
                    print("无法提取有效JSON")
            else:
                print("无有效JSON片段")

if __name__ == "__main__":
    main()