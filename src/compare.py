#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比实验 - 最终修复版
关键修复：正确解析 coverage.analysis2() 的返回值索引
"""

import sys
import json
import random
import string
import os
import time
from datetime import datetime

sys.path.append('.')
sys.path.append('src')

from target_api import app as flask_app
from hybrid_generator import RuleBasedGenerator

# ========== 生成器实现（与之前相同）==========

def random_generate(api_desc, case_types, count_per_type=3):
    """随机生成测试用例"""
    cases = []
    endpoints = [
        ('POST', '/users'),
        ('GET', '/users'),
        ('GET', '/users/1'),
        ('DELETE', '/users/1')
    ]

    for case_type in case_types:
        for i in range(count_per_type):
            method, path = random.choice(endpoints)

            body = None
            if method == 'POST':
                if case_type == 'normal':
                    body = {
                        "username": ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 20))),
                        "age": random.randint(0, 120)
                    }
                elif case_type == 'boundary':
                    boundary_choice = random.choice(['min', 'max'])
                    if boundary_choice == 'min':
                        body = {"username": "abc", "age": 0}
                    else:
                        body = {"username": "a" * 20, "age": 120}
                else:  # exception
                    body = random.choice([
                        {"username": "a", "age": 25},
                        {"username": "test", "age": -1},
                        {"username": None, "age": 25}
                    ])

            request = {
                "method": method,
                "url": path,
                "headers": {"Content-Type": "application/json"} if method == 'POST' else {},
                "body": body
            }

            if case_type == 'normal':
                expected = 201 if method == 'POST' else 200
            elif case_type == 'boundary':
                expected = 201 if method == 'POST' else 200
            else:
                expected = 400

            cases.append({
                "_case_type": case_type,
                "_id": f"random_{case_type}_{i+1}",
                "description": f"随机生成 [{case_type}] {method} {path}",
                "request": request,
                "expected_response": {"status_code": expected}
            })

    return cases


def rule_generate(api_desc, case_types, count_per_type=3):
    """规则生成测试用例"""
    cases = []
    rule_gen = RuleBasedGenerator()

    endpoint_configs = [
        {'method': 'POST', 'path': '/users', 'needs_body': True},
        {'method': 'GET', 'path': '/users', 'needs_body': False},
        {'method': 'GET', 'path': '/users/1', 'needs_body': False},
        {'method': 'DELETE', 'path': '/users/1', 'needs_body': False}
    ]

    for case_type in case_types:
        for i in range(count_per_type):
            config = endpoint_configs[i % len(endpoint_configs)]
            method = config['method']
            path = config['path']

            body = None
            if config['needs_body']:
                if case_type == 'normal':
                    body = {"username": f"user_{i}", "age": 25 + i}
                elif case_type == 'boundary':
                    boundary_choice = i % 2
                    if boundary_choice == 0:
                        body = {"username": "abc", "age": 0}
                    else:
                        body = {"username": "a" * 20, "age": 120}
                else:
                    exception_choice = i % 3
                    if exception_choice == 0:
                        body = {"username": "ab", "age": 25}
                    elif exception_choice == 1:
                        body = {"username": "test", "age": -5}
                    else:
                        body = {"username": None, "age": 25}

            request = {
                "method": method,
                "url": path,
                "headers": {"Content-Type": "application/json"} if config['needs_body'] else {},
                "body": body
            }

            if case_type == 'normal':
                expected = 201 if method == 'POST' else 200
            elif case_type == 'boundary':
                if method == 'POST' and body:
                    username = body.get('username', '')
                    age = body.get('age', 0)
                    if (isinstance(username, str) and len(username) >= 3 and len(username) <= 20 and
                        isinstance(age, int) and age >= 0 and age <= 120):
                        expected = 201
                    else:
                        expected = 400
                else:
                    expected = 200
            else:
                expected = 400

            cases.append({
                "_case_type": case_type,
                "_id": f"rule_{case_type}_{i+1}",
                "description": f"规则生成 [{case_type}] {method} {path}",
                "request": request,
                "expected_response": {"status_code": expected}
            })

    return cases


# ========== 测试执行 ==========

def run_tests_with_client(test_cases, client):
    """执行测试"""
    results = []
    for case in test_cases:
        req = case['request']
        method = req['method']
        path = req['url']
        body = req.get('body')

        start_time = time.time()

        try:
            if method == 'GET':
                resp = client.get(path)
            elif method == 'POST':
                resp = client.post(path, json=body)
            elif method == 'DELETE':
                resp = client.delete(path)
            elif method == 'PUT':
                resp = client.put(path, json=body)
            else:
                resp = None
        except Exception as e:
            results.append({
                'case': case,
                'passed': False,
                'error': str(e),
                'status_code': 500,
                'elapsed': 0
            })
            continue

        elapsed = time.time() - start_time
        expected = case.get('expected_response', {}).get('status_code', 200)
        actual = resp.status_code if resp else 500

        results.append({
            'case': case,
            'passed': actual == expected,
            'status_code': actual,
            'expected': expected,
            'elapsed': round(elapsed, 3)
        })

    return results


# ========== 覆盖率测量（关键修复：正确索引）==========

def measure_coverage(generator_func, name, api_desc, case_types):
    """
    测量覆盖率 - 最终修复版
    正确使用 analysis2 的索引：
    - [1] 可执行行列表 (executable)
    - [3] 未执行行列表 (missing)
    """
    import coverage

    print(f"\n🔄 使用 {name} 生成用例...")
    test_cases = generator_func(api_desc, case_types)
    print(f"   生成 {len(test_cases)} 个用例")

    cov = coverage.Coverage(source=['target_api'])
    cov.start()

    with flask_app.test_client() as client:
        results = run_tests_with_client(test_cases, client)

    cov.stop()
    cov.save()

    total_lines = 0
    missing_lines_list = []

    data = cov.get_data()
    for filename in data.measured_files():
        if 'target_api' in filename:
            # analysis2 返回 (filename, executable, excluded, missing, missing_formatted)
            analysis = cov.analysis2(filename)
            executable_lines = analysis[1]   # 索引1是可执行行
            missing_lines = analysis[3]      # 索引3是缺失行
            total_lines += len(executable_lines)
            missing_lines_list.extend(missing_lines)

    covered_lines = total_lines - len(missing_lines_list)
    coverage_percent = (covered_lines / total_lines * 100) if total_lines > 0 else 0
    passed_count = sum(1 for r in results if r['passed'])

    print("\n📊 Coverage.py原始报告:")
    cov.report(show_missing=True)

    return {
        'name': name,
        'test_cases': len(test_cases),
        'passed': passed_count,
        'failed': len(test_cases) - passed_count,
        'total_lines': total_lines,
        'covered_lines': covered_lines,
        'missing_lines': len(missing_lines_list),
        'coverage': coverage_percent,
        'results': results
    }


def print_results(results):
    """打印结果表格"""
    print("\n" + "="*80)
    print("📈 对比实验结果（最终修复版）")
    print("="*80)
    print(f"{'方法':<10} {'用例数':<8} {'通过':<8} {'失败':<8} {'总行数':<8} {'覆盖行':<8} {'未覆盖':<8} {'覆盖率':<10}")
    print("-"*80)

    for r in results:
        print(f"{r['name']:<10} {r['test_cases']:<8} {r['passed']:<8} {r['failed']:<8} "
              f"{r['total_lines']:<8} {r['covered_lines']:<8} {r['missing_lines']:<8} "
              f"{r['coverage']:.2f}%")

    print("="*80)

    print("\n✅ 数据验证:")
    for r in results:
        check = r['covered_lines'] + r['missing_lines']
        status = "✓" if check == r['total_lines'] else "✗"
        print(f"  {r['name']}: {r['covered_lines']} + {r['missing_lines']} = {check} {status} (应为{r['total_lines']})")


def main():
    api_desc = "POST /users 创建用户，参数：username(string,必填,3-20字符), age(integer,可选,0-120)"
    case_types = ['normal', 'boundary', 'exception']

    print("="*80)
    print("🧪 API测试用例生成方法对比实验（最终修复版）")
    print("="*80)
    print(f"API描述: {api_desc}")
    print(f"用例类型: {case_types}")
    print(f"每种类型生成3个用例，共9个用例/方法")

    results = []

    res_random = measure_coverage(random_generate, 'random', api_desc, case_types)
    results.append(res_random)

    res_rule = measure_coverage(rule_generate, 'rule', api_desc, case_types)
    results.append(res_rule)

    print_results(results)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f'comparison_final_{timestamp}.json'

    save_data = []
    for r in results:
        save_data.append({
            'name': r['name'],
            'test_cases': r['test_cases'],
            'passed': r['passed'],
            'failed': r['failed'],
            'total_lines': r['total_lines'],
            'covered_lines': r['covered_lines'],
            'missing_lines': r['missing_lines'],
            'coverage': r['coverage']
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 结果已保存: {output_file}")


if __name__ == "__main__":
    main()