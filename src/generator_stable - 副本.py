#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳定版测试用例生成器
使用模板生成标准JSON，模型仅用于判断用例类型
"""

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

MODEL_PATH = os.path.abspath("models/fine_tuned_model")
DEVICE = torch.device("cpu")

def load_model():
    """加载微调后的模型"""
    print(f"加载模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE)
    model.eval()
    return tokenizer, model

def parse_input(desc):
    """解析 API 描述，提取方法、路径和描述文本（支持带前缀的输入）"""
    # 先移除可能存在的类型前缀，如 "[NORMAL]"
    desc = re.sub(r'^\[[A-Z]+\]\s*', '', desc)
    match = re.match(r'生成测试用例JSON:\s*(\w+)\s+(/[^\s]+)\s*-\s*(.+)', desc)
    if match:
        return match.group(1), match.group(2), match.group(3)
    # 如果格式不匹配，返回默认值
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
    """用模板生成标准测试用例 JSON，根据生成的值动态确定期望状态码"""

    # 提取路径参数
    path_params = re.findall(r'\{(\w+)\}', path)

    # 根据 case_type 生成路径参数值
    path_param_values = {}
    for param in path_params:
        if case_type == "normal":
            path_param_values[param] = 1          # 存在的用户ID
        else:
            path_param_values[param] = 0           # 不存在的ID

    # 替换 URL 中的占位符
    url = path
    for param, value in path_param_values.items():
        url = url.replace(f'{{{param}}}', str(value))

    # 生成请求体（仅针对 POST）
    body = {}
    if method == "POST":
        if case_type == "normal":
            # 正常值：符合规则
            body = {"username": "testuser", "age": 25}
        elif case_type == "boundary":
            # 边界值：超长字符串和超大年龄
            body = {"username": "a" * 100, "age": 999999}
        else:  # exception
            # 异常值：类型错误
            body = {"username": 12345, "age": "not_a_number"}

    # 根据实际生成的请求确定期望状态码
    if method == "POST":
        if case_type == "normal":
            expected_status = 201
        else:
            expected_status = 400  # 非法数据应返回 400
    elif method == "GET":
        if path_params:
            # 有路径参数，取决于id是否存在
            if case_type == "normal" and path_param_values.get(list(path_params)[0]) == 1:
                expected_status = 200
            else:
                expected_status = 404
        else:
            expected_status = 200
    elif method == "DELETE":
        if path_params:
            if case_type == "normal" and path_param_values.get(list(path_params)[0]) == 1:
                expected_status = 204
            else:
                expected_status = 404
        else:
            expected_status = 204
    else:
        # 其他方法暂默认200
        expected_status = 200

    # 组装最终 JSON
    test_case = {
        "description": f"{method} {path} - {case_type} test",
        "case_type": case_type,
        "request": {
            "method": method,
            "url": url,
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            "query": {},
            "body": body if method in ["POST", "PUT", "PATCH"] else {}
        },
        "expected_response": {
            "status_code": expected_status,
            "headers": {
                "Content-Type": "application/json"
            }
        }
    }

    # 添加响应体示例（仅用于正常的 GET 请求）
    if method == "GET" and case_type == "normal" and expected_status == 200:
        test_case["expected_response"]["body"] = {
            "id": 1,
            "username": "preloaded",
            "age": 30
        }
    elif case_type != "normal":
        # 错误响应的示例
        test_case["expected_response"]["body"] = {
            "error": "Invalid request",
            "code": "VALIDATION_ERROR"
        }

    return test_case

def generate_test_case(api_description, tokenizer, model):
    """主生成函数：调用模型判断类型，然后生成模板JSON"""
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

# 可选测试代码
if __name__ == "__main__":
    tokenizer, model = load_model()
    test_descriptions = [
        "生成测试用例JSON: POST /users - 创建用户，需要username（字符串，必填，3-20字符）和age（整数，可选，0-120）",
        "生成测试用例JSON: GET /users/{id} - 获取用户信息，id为整数",
        "生成测试用例JSON: DELETE /users/{id} - 删除用户，id为整数",
    ]
    for desc in test_descriptions:
        print(f"\n输入: {desc}")
        result = generate_test_case(desc, tokenizer, model)
        print(json.dumps(result, indent=2, ensure_ascii=False))