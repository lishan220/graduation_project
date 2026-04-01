import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

MODEL_PATH = os.path.abspath("models/fine_tuned_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    print(f"加载本地模型从 {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
    return tokenizer, model

def extract_and_rebuild_json(raw_text, original_desc):
    """
    💡 V4 终极版：放弃修补破损字符串，转而提取核心语义信息，进行模板化重组。
    这保证了 100% 输出合法 JSON。
    """
    # 1. 准备标准测试用例模板
    test_case = {
        "description": original_desc,
        "request": {
            "method": "GET",  # 默认值
            "url": "",
            "headers": {},
            "body": {},
            "query": {}
        },
        "expected_response": {
            "status_code": 200
        },
        "_generation_method": "t5_with_slot_filling_rebuild"
    }

    # 2. 提取 Method
    method_match = re.search(r'"method"\s*:\s*"([A-Z]+)"', raw_text)
    if method_match:
        test_case["request"]["method"] = method_match.group(1)

    # 3. 提取 URL
    url_match = re.search(r'"url"\s*:\s*"([^"]+)"', raw_text)
    if url_match:
        test_case["request"]["url"] = url_match.group(1)

    # 4. 提取 Status Code
    status_match = re.search(r'"status_code"\s*:\s*(\d+)', raw_text)
    if status_match:
        test_case["expected_response"]["status_code"] = int(status_match.group(1))

    # 5. 尝试提取 Description 描述片段
    desc_match = re.search(r'(normal case \d+|boundary case \d+|exception case \d+)', raw_text, re.IGNORECASE)
    if desc_match:
        test_case["description"] += f" ({desc_match.group(1)})"

    return test_case, None

def generate_test_case_pure_model(api_description, tokenizer, model, max_length=1024):
    inputs = tokenizer(api_description, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length, 
            num_beams=4,
            early_stopping=True,
            do_sample=False
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 使用“语义提取+重组”策略
    parsed_json, _ = extract_and_rebuild_json(decoded, api_description)
    
    # 记录原始输出用于对比分析
    parsed_json["_raw_model_output"] = decoded 
    
    return parsed_json

def main():
    tokenizer, model = load_model()
    
    test_descriptions = [
       "[NORMAL] POST /users - 创建用户，需要username（字符串，必填，3-20字符）和age（整数，可选，0-120）",
       "[BOUNDARY] GET /pet/{petId} - 通过ID获取宠物信息，petId为整数",
       "[EXCEPTION] DELETE /users/{id} - 根据ID删除用户"
    ]
    
    for desc in test_descriptions:
        print("\n" + "="*60)
        print(f"输入: {desc}")
        result = generate_test_case_pure_model(desc, tokenizer, model)
        print(f"生成:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
        print("="*60)

if __name__ == "__main__":
    main()