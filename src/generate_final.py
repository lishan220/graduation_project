import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

MODEL_PATH = os.path.abspath("models/fine_tuned_model")
DEVICE = torch.device("cpu")

def load_model():
    print(f"加载模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE)
    model.eval()
    return tokenizer, model

def parse_input(desc):
    """解析 API 描述"""
    match = re.match(r'生成测试用例JSON:\s*(\w+)\s+(/[^\s]+)\s*-\s*(.+)', desc)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return "GET", "/unknown", "test"

def detect_case_type(raw_output):
    """从模型输出中检测测试类型"""
    raw_lower = raw_output.lower()
    if "boundary" in raw_lower or "边界" in raw_lower:
        return "boundary"
    elif "exception" in raw_lower or "异常" in raw_lower or "error" in raw_lower:
        return "exception"
    else:
        return "normal"

def generate_test_json(method, path, description, case_type):
    """用模板生成标准测试用例 JSON"""
    
    # 状态码映射
    status_map = {
        "GET": 200,
        "POST": 201, 
        "PUT": 200,
        "DELETE": 204,
        "PATCH": 200
    }
    base_status = status_map.get(method, 200)
    
    # 根据 case_type 调整
    if case_type == "exception":
        status_code = 400 if method != "DELETE" else 404
    else:
        status_code = base_status
    
    # 提取路径参数
    path_params = re.findall(r'\{(\w+)\}', path)
    
    # 从描述中提取字段信息（简化版）
    body = {}
    fields = re.findall(r'(\w+)\s*（([^）]+)）', description)
    
    for field_name, field_desc in fields:
        # 判断字段类型
        field_type = "string"
        if "整数" in field_desc or "int" in field_desc:
            field_type = "integer"
        elif "布尔" in field_desc:
            field_type = "boolean"
        
        # 生成测试值
        is_required = "必填" in field_desc
        
        if case_type == "boundary":
            # 边界值测试
            if field_type == "string":
                # 超长字符串或空字符串
                body[field_name] = "a" * 100 if is_required else ""
            elif field_type == "integer":
                # 超大值或负数
                body[field_name] = 999999
            else:
                body[field_name] = None
        elif case_type == "exception":
            # 异常测试 - 错误类型
            if field_type == "string":
                body[field_name] = 12345  # 类型错误
            elif field_type == "integer":
                body[field_name] = "not_a_number"
            else:
                body[field_name] = None
        else:
            # normal - 正常值
            if field_type == "string":
                # 提取长度限制
                len_match = re.search(r'(\d+)-(\d+)', field_desc)
                if len_match:
                    min_len = int(len_match.group(1))
                    body[field_name] = "a" * min_len
                else:
                    body[field_name] = f"test_{field_name}"
            elif field_type == "integer":
                range_match = re.search(r'(\d+)-(\d+)', field_desc)
                if range_match:
                    min_val = int(range_match.group(1))
                    body[field_name] = min_val
                else:
                    body[field_name] = 1
            else:
                body[field_name] = True
    
    # 组装最终 JSON
    test_case = {
        "description": f"{method} {path} - {case_type} test",
        "case_type": case_type,
        "request": {
            "method": method,
            "url": f"http://api.example.com{path}",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            "path_params": {p: f"test_{p}_value" for p in path_params} if path_params else {},
            "query": {},
            "body": body if method in ["POST", "PUT", "PATCH"] else {}
        },
        "expected_response": {
            "status_code": status_code,
            "headers": {
                "Content-Type": "application/json"
            }
        }
    }
    
    # 添加响应体（根据类型）
    if case_type == "normal" and method == "GET":
        test_case["expected_response"]["body"] = {
            "id": "test_id",
            "type": "object"
        }
    elif case_type == "exception":
        test_case["expected_response"]["body"] = {
            "error": "Invalid request",
            "code": "VALIDATION_ERROR"
        }
    
    return test_case

def generate_test_case(api_description, tokenizer, model):
    """主生成函数"""
    
    # 解析输入
    method, path, description = parse_input(api_description)
    
    # 调用模型（只用来判断 case_type）
    inputs = tokenizer(api_description, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            num_beams=3,
            early_stopping=True
        )
    
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   模型原始输出: {raw[:80]}...")
    
    # 检测类型
    case_type = detect_case_type(raw)
    print(f"   检测到类型: {case_type}")
    
    # 用模板生成标准 JSON
    return generate_test_json(method, path, description, case_type)

def main():
    tokenizer, model = load_model()
    
    test_cases = [
        "生成测试用例JSON: POST /users - 创建用户，需要username（字符串，必填，3-20字符）和age（整数，可选，0-120）",
        "生成测试用例JSON: GET /pet/{petId} - 通过ID获取宠物信息，petId为整数",
        "生成测试用例JSON: DELETE /users/{id} - 根据ID删除用户",
        "生成测试用例JSON: PUT /products/{productId} - 更新产品信息，需要name（字符串，可选）和price（数字，必填，>0）",
    ]
    
    for desc in test_cases:
        print(f"\n{'='*70}")
        print(f"输入: {desc[:60]}...")
        result = generate_test_case(desc, tokenizer, model)
        print(f"\n生成结果:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
        print('='*70)

if __name__ == "__main__":
    main()