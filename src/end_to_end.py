#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端演示：先生成测试用例，再执行测试，最后生成报告
"""
import json
import sys
from generator_stable import load_model, generate_test_case
from test_runner import TestRunner

def main():
    # 1. 加载模型
    print("加载模型...")
    tokenizer, model = load_model()

    # 2. 定义API描述（只使用 target_api 支持的端点）
    api_descriptions = [
        "生成测试用例JSON: POST /users - 创建用户，需要username（字符串，必填，3-20字符）和age（整数，可选，0-120）",
        "生成测试用例JSON: GET /users/{id} - 获取用户信息，id为整数",
        "生成测试用例JSON: DELETE /users/{id} - 删除用户，id为整数",
    ]

    # 3. 生成测试用例（每个描述生成三类用例）
    test_cases = []
    case_types = ['normal', 'boundary', 'exception']

    for desc in api_descriptions:
        for case_type in case_types:
            typed_desc = f"[{case_type.upper()}] {desc}"
            case = generate_test_case(typed_desc, tokenizer, model)
            case['_case_type'] = case_type
            test_cases.append(case)

    # 保存生成的用例
    with open('data/generated_cases.json', 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)
    print(f"已生成 {len(test_cases)} 个测试用例，保存至 data/generated_cases.json")

    # 4. 执行测试（指向本地服务）
    runner = TestRunner(base_url='http://localhost:5000')
    results = runner.run(test_cases)
    report_path = runner.generate_html_report('reports/end_to_end_report.html')
    print(f"端到端测试完成！报告: {report_path}")

if __name__ == "__main__":
    main()