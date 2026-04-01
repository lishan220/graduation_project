import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# 使用绝对路径，并添加 local_files_only=True 强制只加载本地文件
MODEL_PATH = os.path.abspath("models/fine_tuned_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    print(f"加载本地模型从 {MODEL_PATH} ...")
    # 检查路径是否存在
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型路径不存在: {MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE)
    model.eval()
    return tokenizer, model

def generate_test_case(api_description, tokenizer, model, max_length=256):
    inputs = tokenizer(api_description, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
        do_sample=False,
        repetition_penalty=1.2,
        length_penalty=2.0
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        test_case = json.loads(decoded)
        return test_case
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', decoded, re.DOTALL)
        if match:
            try:
                test_case = json.loads(match.group())
                return test_case
            except:
                pass
        return {"error": "生成的输出不是有效JSON", "raw": decoded}
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
        result = generate_test_case(desc, tokenizer, model)
        print(f"生成: {json.dumps(result, indent=2, ensure_ascii=False)}")
        print("="*60)

if __name__ == "__main__":
    main()