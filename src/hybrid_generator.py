#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版测试用例生成器（简化置信度）
模型参与：类型判断 + 参数提取 + 字段推断
"""

import json
import re
import torch
import hashlib
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
    """解析 API 描述，提取方法、路径、描述文本及约束。"""
    desc = re.sub(r'^\[[A-Z]+\]\s*', '', desc)
    patterns = [
        r'(?:生成测试用例JSON:)?\s*(\w+)\s+(/[^\s-]+)\s*-\s*(.+)',
        r'(?:生成测试用例JSON:)?\s*(\w+)\s+(/[^\s]+)\s+(.+)',
        r'(?:生成测试用例JSON:)?\s*(\w+)\s+(/[^\s]+)',
    ]
    method, path, desc_text = "GET", "/unknown", "test"
    for pattern in patterns:
        match = re.match(pattern, desc, re.IGNORECASE)
        if match:
            method = match.group(1).upper()
            path = match.group(2)
            desc_text = match.group(3) if len(match.groups()) > 2 else ""
            break

    constraints = {}
    # 字段名
    fields = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)[（\(]', desc_text)
    # 过滤掉常见的非字段词
    stop_fields = ['需要', '和', '例如', '参数', '必填', '可选', '字符串', '整数']
    fields = [f for f in fields if f not in stop_fields and not f.isdigit()]
    constraints['fields'] = fields
    # 长度范围
    len_range = re.search(r'(\d+)[-~](\d+)\s*字符', desc_text)
    if len_range:
        constraints['min_length'] = int(len_range.group(1))
        constraints['max_length'] = int(len_range.group(2))
    else:
        max_len = re.search(r'(\d+)\s*字符', desc_text)
        if max_len:
            constraints['max_length'] = int(max_len.group(1))
        constraints['min_length'] = constraints.get('min_length', 1)
    # 数值范围
    num_range = re.search(r'(\d+)[-~](\d+)(?!\s*字符)', desc_text)
    if num_range:
        constraints['min_value'] = int(num_range.group(1))
        constraints['max_value'] = int(num_range.group(2))
    else:
        constraints['min_value'] = constraints.get('min_value', 0)
        constraints['max_value'] = constraints.get('max_value', 120)
    # 必填/可选字段
    constraints['required_fields'] = re.findall(r'(\w+)[（\(][^）)]*必填[^）)]*[）\)]', desc_text)
    constraints['optional_fields'] = re.findall(r'(\w+)[（\(][^）)]*可选[^）)]*[）\)]', desc_text)
    # 字段类型推断
    constraints['field_types'] = {}
    for field in fields:
        if re.search(rf'{field}.*(?:整数|数字|int|age|id)', desc_text, re.IGNORECASE):
            constraints['field_types'][field] = 'integer'
        elif re.search(rf'{field}.*(?:字符串|字符|string|str|username|name)', desc_text, re.IGNORECASE):
            constraints['field_types'][field] = 'string'
        elif re.search(rf'{field}.*(?:布尔|bool)', desc_text, re.IGNORECASE):
            constraints['field_types'][field] = 'boolean'
        else:
            constraints['field_types'][field] = 'string'
    return method, path, desc_text, constraints

def model_inference(api_description, tokenizer, model, num_sequences=3, num_beams=3):
    """生成多条候选输出，返回 (decoded_list, confidences)  confidences 暂设为全0"""
    inputs = tokenizer(
        api_description,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,
            num_beams=num_beams,
            num_return_sequences=num_sequences,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=False,   # 关闭分数输出，避免索引错误
        )

    sequences = outputs.sequences
    decoded = [tokenizer.decode(seq, skip_special_tokens=True) for seq in sequences]
    confidences = [0.0] * num_sequences   # 占位，不实际计算
    return decoded, confidences

def extract_parameters(text):
    """从模型输出中提取参数建议"""
    params = {}
    low = text.lower()
    # username 长度
    for pat in [r'username.*?length.*?(\d+)', r'username.*?(\d+).*?字符', r'username.*?(\d+)']:
        match = re.search(pat, low)
        if match:
            params['username_len'] = int(match.group(1))
            break
    # age 值
    for pat in [r'age.*?value.*?(\d+)', r'age.*?(\d+).*?岁', r'age.*?(\d+)']:
        match = re.search(pat, low)
        if match:
            params['age_value'] = int(match.group(1))
            break
    # 状态码
    match = re.search(r'\b(200|201|204|400|404|422|500)\b', text)
    if match:
        params['suggested_status'] = int(match.group(1))
    # 类型提示
    if any(kw in low for kw in ['boundary', '边界', 'edge', 'limit']):
        params['detected_type_hint'] = 'boundary'
    elif any(kw in low for kw in ['exception', '异常', 'error', 'invalid']):
        params['detected_type_hint'] = 'exception'
    else:
        params['detected_type_hint'] = 'normal'
    return params

def generate_field_values(case_type, constraints, extracted):
    """根据测试类型和约束生成字段值"""
    suggestions = {}
    max_len = constraints.get('max_length', 20)
    min_len = constraints.get('min_length', 3)
    max_val = constraints.get('max_value', 120)
    min_val = constraints.get('min_value', 0)
    field_types = constraints.get('field_types', {})

    for field in constraints.get('fields', []):
        ftype = field_types.get(field, 'string')
        if case_type == 'normal':
            if ftype == 'integer':
                suggestions[field] = (max_val + min_val) // 2
            elif ftype == 'string':
                mid_len = (max_len + min_len) // 2
                suggestions[field] = 't' * max(mid_len, 1)
            else:
                suggestions[field] = True
        elif case_type == 'boundary':
            h = int(hashlib.md5(field.encode()).hexdigest(), 16)
            if ftype == 'integer':
                suggestions[field] = max_val if h % 2 == 0 else min_val
            elif ftype == 'string':
                suggestions[field] = 'a' * (max_len if h % 2 == 0 else min_len)
            else:
                suggestions[field] = True
        else:  # exception
            if ftype == 'integer':
                h = int(hashlib.md5(field.encode()).hexdigest(), 16)
                exceptions = ["not_a_number", None, -99999, max_val + 1000]
                suggestions[field] = exceptions[h % len(exceptions)]
            elif ftype == 'string':
                h = int(hashlib.md5(field.encode()).hexdigest(), 16)
                exceptions = [None, "", "a" * (max_len + 100), "<script>alert(1)</script>"]
                suggestions[field] = exceptions[h % len(exceptions)]
            else:
                suggestions[field] = "not_boolean"
    return suggestions

def assemble_test_case(method, path, desc_text, analysis, constraints):
    """用模型分析结果构建最终 JSON"""
    case_type = analysis["case_type"]
    field_suggestions = analysis["field_suggestions"]
    extracted = analysis["extracted_params"]

    # 路径参数处理
    path_params = re.findall(r'\{(\w+)\}', path)
    url = path
    for param in path_params:
        if case_type == "normal":
            val = 1
        elif case_type == "boundary":
            val = 999999
        else:
            val = -1
        url = url.replace(f'{{{param}}}', str(val))

    # 请求体
    body = {}
    if method in ["POST", "PUT", "PATCH"]:
        if field_suggestions:
            body = field_suggestions
        else:
            body = {
                "normal": {"username": "testuser", "age": 25},
                "boundary": {"username": "a" * extracted.get('username_len', 100),
                             "age": extracted.get('age_value', 999999)},
                "exception": {"username": None, "age": "not_a_number"}
            }.get(case_type, {})

    # 期望状态码
    if "suggested_status" in extracted:
        expected_status = extracted["suggested_status"]
    else:
        if case_type == "normal":
            if method == "POST":
                expected_status = 201
            elif method == "DELETE":
                expected_status = 204 if path_params else 200
            else:
                expected_status = 200
        else:
            if method == "POST":
                expected_status = 400
            elif method in ["GET", "DELETE"] and path_params:
                expected_status = 404 if case_type == "exception" else 400
            else:
                expected_status = 400

    test_case = {
        "description": f"{method} {path} - {case_type} test",
        "case_type": case_type,
        "model_analysis": {
            "extracted_params": extracted,
            "field_suggestions": field_suggestions,
        },
        "request": {
            "method": method,
            "url": url,
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            "query": {},
            "body": body
        },
        "expected_response": {
            "status_code": expected_status,
            "headers": {"Content-Type": "application/json"}
        }
    }

    if method == "GET" and case_type == "normal" and expected_status == 200:
        test_case["expected_response"]["body"] = {
            "id": 1, "username": "preloaded", "age": 30
        }
    elif case_type != "normal":
        test_case["expected_response"]["body"] = {
            "error": "Invalid request",
            "code": "VALIDATION_ERROR"
        }

    return test_case

def analyze_model_output(raw_outputs, confidences, constraints):
    """分析候选输出，投票决定类型并提取参数"""
    # 简单投票（每票权重1）
    type_votes = {"boundary": 0, "exception": 0, "normal": 0}
    type_keywords = {
        "boundary": ["boundary", "边界", "edge", "limit", "max", "min", "极限"],
        "exception": ["exception", "异常", "error", "invalid", "wrong", "错误", "非法"],
        "normal": ["normal", "正常", "standard", "valid", "正确"]
    }
    for txt in raw_outputs:
        low = txt.lower()
        for ttype, kws in type_keywords.items():
            if any(kw in low for kw in kws):
                type_votes[ttype] += 1
                break
        else:
            type_votes["normal"] += 0.5

    case_type = max(type_votes, key=type_votes.get)
    # 取第一条候选提取参数（可优化为取投票获胜的那条，但简化版直接用第一条）
    best_output = raw_outputs[0]
    extracted = extract_parameters(best_output)
    field_suggestions = generate_field_values(case_type, constraints, extracted)

    return {
        "case_type": case_type,
        "type_votes": type_votes,
        "extracted_params": extracted,
        "field_suggestions": field_suggestions,
        "best_raw_output": best_output,
    }

def generate_test_case(api_description, tokenizer, model):
    """主生成函数（对外接口）"""
    method, path, desc_text, constraints = parse_input(api_description)
    print(f"\n   [解析] 方法={method}, 路径={path}")
    print(f"   [解析] 约束={constraints}")

    raw_outputs, _ = model_inference(api_description, tokenizer, model, num_sequences=3, num_beams=3)
    print(f"   [模型] 候选输出:")
    for i, txt in enumerate(raw_outputs):
        print(f"          候选{i+1}: {txt[:80]}")

    analysis = analyze_model_output(raw_outputs, None, constraints)
    print(f"   [分析] 类型投票={analysis['type_votes']}")
    print(f"   [分析] 选中类型={analysis['case_type']}")
    print(f"   [分析] 字段推断={analysis['field_suggestions']}")

    return assemble_test_case(method, path, desc_text, analysis, constraints)

if __name__ == "__main__":
    tokenizer, model = load_model()
    test_descriptions = [
        "生成测试用例JSON: POST /users - 创建用户，需要username（字符串，必填，3-20字符）和age（整数，可选，0-120）",
        "生成测试用例JSON: GET /users/{id} - 获取用户信息，id为整数",
        "生成测试用例JSON: DELETE /users/{id} - 删除用户，id为整数",
    ]
    for desc in test_descriptions:
        print("\n" + "=" * 70)
        print(f"输入: {desc}")
        result = generate_test_case(desc, tokenizer, model)
        print("\n── 生成结果 ──")
        print(json.dumps(result, indent=2, ensure_ascii=False))