#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端演示：先生成测试用例，再执行测试，最后生成报告
"""
import json
import sys
from hybrid_generator import load_model, RuleBasedGenerator, generate_test_case
from test_runner import TestRunner

def main():
    # 1. 加载模型和规则生成器
    print("加载模型...")
    tokenizer, model = load_model()
    rule_gen = RuleBasedGenerator()

    # 2. 定义API描述
    api_descriptions = [
        "生成测试用例JSON: POST /users - 创建用户，需要username（字符串，必填，3-20字符）和age（整数，可选，0-120）",
        "生成测试用例JSON: GET /pet/{petId} - 通过ID获取宠物信息，petId为整数",
        "生成测试用例JSON: DELETE /users/{id} - 根据ID删除用户",
    ]

    # 3. 生成测试用例（每个描述生成三类用例）
    test_cases = []
    case_types = ['normal', 'boundary', 'exception']

    for desc in api_descriptions:
        for case_type in case_types:
            typed_desc = f"[{case_type.upper()}] {desc}"
            # 注意：传递 case_type 参数
            case = generate_test_case(typed_desc, tokenizer, model, rule_gen, case_type=case_type)
            case.pop('_source', None)
            case['_case_type'] = case_type
            test_cases.append(case)

    # 保存生成的用例
    with open('data/generated_cases.json', 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)
    print(f"已生成 {len(test_cases)} 个测试用例，保存至 data/generated_cases.json")

    # 4. 执行测试
    runner = TestRunner(base_url='https://httpbin.org')
    results = runner.run(test_cases)
    report_path = runner.generate_html_report('reports/end_to_end_report.html')
    print(f"端到端测试完成！报告: {report_path}")

if __name__ == "__main__":
    main()