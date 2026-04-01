# 生成优化后的 compare_v3.py 代码

code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比实验 - 优化版
对比三种方法：随机生成、规则生成、模型生成
优化点：
1. 规则生成也覆盖多个端点（GET/DELETE）
2. 修正覆盖率显示，使用 Coverage.py 的原始报告
3. 优化预期状态码判断逻辑
"""

import sys
import json
import random
import string
import os
import time
import re
from io import StringIO

# 设置环境变量避免警告
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

sys.path.append('.')
sys.path.append('src')

from target_api import app as flask_app

# 尝试导入模型生成器
try:
    from generate_test import load_model, generate_hybrid
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("⚠️ 模型生成器未找到，将只对比随机和规则方法")

# ========== 生成器实现 ==========

def random_generate(api_desc, case_types, count_per_type=3):
    """
    随机生成测试用例
    覆盖多个端点：POST /users, GET /users, GET /users/<id>, DELETE /users/<id>
    """
    cases = []
    endpoints = [
        ('POST', '/users'),
        ('GET', '/users'),
        ('GET', '/users/{id}'),
        ('DELETE', '/users/{id}')
    ]
    
    for case_type in case_types:
        for i in range(count_per_type):
            # 随机选择端点
            method, path_template = random.choice(endpoints)
            
            # 处理路径参数
            path = path_template
            if '{id}' in path:
                path = path.format(id=random.randint(1, 10))
            
            # 生成请求体
            body = None
            if method == 'POST':
                if case_type == 'normal':
                    body = {
                        "username": ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 20))),
                        "age": random.randint(0, 120)
                    }
                elif case_type == 'boundary':
                    boundary_choice = random.choice(['min', 'max', 'under', 'over'])
                    if boundary_choice == 'min':
                        body = {"username": "abc", "age": 0}
                    elif boundary_choice == 'max':
                        body = {"username": "a" * 20, "age": 120}
                    elif boundary_choice == 'under':
                        body = {"username": "ab", "age": -1}
                    else:
                        body = {"username": "a" * 21, "age": 121}
                else:  # exception
                    body = random.choice([
                        {"username": None, "age": 25},
                        {"username": "test", "age": "not_a_number"},
                        {"username": "<script>alert(1)</script>", "age": 25},
                        {"unexpected_field": "value"},
                        {}
                    ])
            
            request = {
                "method": method,
                "url": path,
                "headers": {"Content-Type": "application/json"} if method == 'POST' else {},
                "body": body
            }
            
            # 更准确的预期状态码
            expected = calculate_expected_status(method, path, body, case_type)
            
            cases.append({
                "_case_type": case_type,
                "_id": f"random_{case_type}_{i+1}",
                "description": f"随机生成 [{case_type}] {method} {path}",
                "request": request,
                "expected_response": {"status_code": expected}
            })
    
    return cases


def rule_generate(api_desc, case_types, count_per_type=3):
    """
    基于规则的生成 - 优化版
    同样覆盖多个端点，但按规则系统性生成数据
    """
    cases = []
    
    # 定义所有端点的规则
    endpoint_rules = [
        {
            'method': 'POST',
            'path': '/users',
            'params': {
                'username': {'type': 'string', 'min': 3, 'max': 20, 'required': True},
                'age': {'type': 'integer', 'min': 0, 'max': 120, 'required': False}
            }
        },
        {
            'method': 'GET',
            'path': '/users',
            'params': {}
        },
        {
            'method': 'GET',
            'path': '/users/{id}',
            'params': {'id': {'type': 'integer', 'min': 1, 'max': 100}}
        },
        {
            'method': 'DELETE',
            'path': '/users/{id}',
            'params': {'id': {'type': 'integer', 'min': 1, 'max': 100}}
        }
    ]
    
    for case_type in case_types:
        for i in range(count_per_type):
            # 循环使用不同端点，确保覆盖全面
            rule = endpoint_rules[i % len(endpoint_rules)]
            method = rule['method']
            path_template = rule['path']
            
            # 处理路径参数
            path = path_template
            if '{id}' in path:
                if case_type == 'normal':
                    path = path.format(id=random.randint(1, 5))  # 假设1-5存在
                elif case_type == 'boundary':
                    path = path.format(id=random.choice([1, 100]))  # 边界ID
                else:
                    path = path.format(id=random.choice([9999, -1, 0]))  # 异常ID
            
            # 生成请求体
            body = None
            if method == 'POST' and 'params' in rule and rule['params']:
                body = {}
                username_rule = rule['params']['username']
                age_rule = rule['params'].get('age')
                
                if case_type == 'normal':
                    # 正常值：范围内随机
                    username_len = random.randint(username_rule['min'], username_rule['max'])
                    body['username'] = 'u' * username_len
                    if age_rule:
                        body['age'] = random.randint(age_rule['min'], age_rule['max'])
                        
                elif case_type == 'boundary':
                    # 边界值：最小、最大、刚好超出边界
                    boundary_choice = i % 4
                    if boundary_choice == 0:
                        body['username'] = 'a' * username_rule['min']  # 最小长度
                        body['age'] = age_rule['min'] if age_rule else 0
                    elif boundary_choice == 1:
                        body['username'] = 'a' * username_rule['max']  # 最大长度
                        body['age'] = age_rule['max'] if age_rule else 120
                    elif boundary_choice == 2:
                        body['username'] = 'a' * (username_rule['min'] - 1)  # 小于最小
                        body['age'] = age_rule['min'] - 1 if age_rule else -1
                    else:
                        body['username'] = 'a' * (username_rule['max'] + 1)  # 大于最大
                        body['age'] = age_rule['max'] + 1 if age_rule else 121
                        
                else:  # exception
                    # 异常值：类型错误、空值、特殊字符
                    exception_choice = i % 4
                    if exception_choice == 0:
                        body['username'] = None
                        body['age'] = 25
                    elif exception_choice == 1:
                        body['username'] = 12345  # 类型错误
                        body['age'] = "twenty-five"
                    elif exception_choice == 2:
                        body['username'] = "'; DROP TABLE users; --"  # SQL注入
                        body['age'] = 25
                    else:
                        body['username'] = "<img src=x onerror=alert('xss')>"  # XSS
                        body['age'] = 25
            
            request = {
                "method": method,
                "url": path,
                "headers": {"Content-Type": "application/json"} if method == 'POST' else {},
                "body": body
            }
            
            # 计算预期状态码
            expected = calculate_expected_status(method, path, body, case_type)
            
            cases.append({
                "_case_type": case_type,
                "_id": f"rule_{case_type}_{i+1}",
                "description": f"规则生成 [{case_type}] {method} {path}",
                "request": request,
                "expected_response": {"status_code": expected}
            })
    
    return cases


def calculate_expected_status(method, path, body, case_type):
    """
    根据请求计算预期状态码
    更智能的判断逻辑
    """
    # POST /users 创建用户
    if method == 'POST' and path == '/users':
        if case_type == 'normal':
            return 201
        elif case_type == 'boundary':
            # 检查边界值是否符合约束
            if body and isinstance(body.get('username'), str):
                username_len = len(body['username'])
                age = body.get('age')
                
                # 用户名长度在3-20之间，年龄在0-120之间（或不存在）
                username_valid = 3 <= username_len <= 20
                age_valid = (age is None) or (isinstance(age, int) and 0 <= age <= 120)
                
                return 201 if (username_valid and age_valid) else 400
            return 400
        else:  # exception
            return 400
    
    # GET /users 列表
    if method == 'GET' and path == '/users':
        return 200
    
    # GET /users/{id} 获取单个用户
    if method == 'GET' and re.match(r'/users/\\d+', path):
        if case_type == 'exception':
            # 异常ID应该返回404
            return 404
        # 正常ID可能返回200或404（如果不存在）
        return 200
    
    # DELETE /users/{id}
    if method == 'DELETE' and re.match(r'/users/\\d+', path):
        if case_type == 'exception':
            return 404
        return 204  # No Content
    
    return 200


def model_generate(api_desc, case_types, count_per_type=3):
    """基于深度学习模型的生成"""
    if not MODEL_AVAILABLE:
        print("⚠️ 模型不可用，跳过模型生成")
        return []
    
    cases = []
    
    try:
        tokenizer, model, _ = load_model()
        
        for case_type in case_types:
            for i in range(count_per_type):
                typed_desc = f"[{case_type.upper()}] {api_desc}"
                result = generate_hybrid(typed_desc, tokenizer, model, case_type=case_type)
                
                if 'request' in result:
                    # 修正URL为本地路径
                    result['request']['url'] = '/users'
                    result['_case_type'] = case_type
                    result['_id'] = f"model_{case_type}_{i+1}"
                    # 重新计算预期状态码
                    req = result['request']
                    result['expected_response']['status_code'] = calculate_expected_status(
                        req.get('method', 'POST'),
                        req.get('url', '/users'),
                        req.get('body'),
                        case_type
                    )
                    cases.append(result)
                
    except Exception as e:
        print(f"⚠️ 模型生成失败: {e}")
    
    return cases


# ========== 测试执行 ==========

def run_single_test(case, client):
    """执行单个测试用例"""
    req = case['request']
    method = req['method']
    path = req['url']
    headers = req.get('headers', {})
    body = req.get('body')
    
    start_time = time.time()
    
    try:
        if method == 'GET':
            resp = client.get(path, headers=headers)
        elif method == 'POST':
            resp = client.post(path, json=body, headers=headers)
        elif method == 'DELETE':
            resp = client.delete(path, headers=headers)
        elif method == 'PUT':
            resp = client.put(path, json=body, headers=headers)
        else:
            return {'passed': False, 'error': 'Unknown method', 'status_code': 500, 'elapsed': 0}
    except Exception as e:
        return {'passed': False, 'error': str(e), 'status_code': 500, 'elapsed': time.time() - start_time}
    
    elapsed = time.time() - start_time
    expected = case.get('expected_response', {}).get('status_code', 200)
    actual = resp.status_code
    
    return {
        'passed': actual == expected,
        'status_code': actual,
        'expected': expected,
        'response_body': resp.get_data(as_text=True)[:500],
        'elapsed': round(elapsed, 3)
    }


def run_all_tests(test_cases, client):
    """运行所有测试用例"""
    results = []
    for case in test_cases:
        result = run_single_test(case, client)
        results.append({'case': case, 'result': result})
    return results


# ========== 覆盖率测量 ==========

def measure_coverage(generator_name, generator_func, api_desc, case_types):
    """
    测量指定生成器的覆盖率
    使用 Coverage.py 的原始报告数据
    """
    import coverage
    
    # 1. 生成测试用例
    print(f"\\n🔄 使用 {generator_name} 生成用例...")
    test_cases = generator_func(api_desc, case_types, count_per_type=3)
    print(f"   生成 {len(test_cases)} 个用例")
    
    # 显示用例分布
    type_count = {}
    endpoint_count = {}
    for c in test_cases:
        ct = c['_case_type']
        ep = f"{c['request']['method']} {c['request']['url']}"
        type_count[ct] = type_count.get(ct, 0) + 1
        endpoint_count[ep] = endpoint_count.get(ep, 0) + 1
    
    print(f"   类型分布: {type_count}")
    print(f"   端点分布: {dict(list(endpoint_count.items())[:5])}")  # 只显示前5个
    
    # 2. 创建全新的coverage实例
    cov = coverage.Coverage(
        source=['target_api'],
        omit=['*/venv/*', '*/test_*', '*/__pycache__/*', '*/src/compare_*'],
        branch=True
    )
    cov.erase()
    cov.start()
    
    # 3. 执行测试
    with flask_app.test_client() as client:
        results = run_all_tests(test_cases, client)
    
    cov.stop()
    cov.save()
    
    # 4. 获取 Coverage.py 的原始报告数据
    # 方法1：从分析数据获取
    analysis = cov.analysis('target_api.py')
    filename, executable_lines, missing_lines, excluded_lines = analysis
    
    total_statements = len(executable_lines)
    missing_statements = len(missing_lines)
    covered_statements = total_statements - missing_statements
    
    # 方法2：获取分支覆盖率
    try:
        branch_coverage = cov.branch_coverage()
    except:
        branch_coverage = None
    
    # 方法3：解析report()输出获取综合覆盖率
    output = StringIO()
    cov.report(file=output, show_missing=True)
    report_lines = output.getvalue().split('\\n')
    
    # 解析最后一行TOTAL获取Coverage.py计算的覆盖率
    coverage_py_percent = None
    for line in report_lines:
        if line.startswith('TOTAL'):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    coverage_py_percent = float(parts[-1].replace('%', ''))
                except:
                    pass
    
    # 如果没有TOTAL行，尝试从文件行解析
    if coverage_py_percent is None:
        for line in report_lines:
            if 'target_api.py' in line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        coverage_py_percent = float(parts[-1].replace('%', ''))
                    except:
                        pass
                break
    
    # 5. 统计测试结果
    passed_count = sum(1 for r in results if r['result']['passed'])
    
    return {
        'name': generator_name,
        'test_cases': len(test_cases),
        'passed': passed_count,
        'failed': len(test_cases) - passed_count,
        'total_statements': total_statements,
        'covered_statements': covered_statements,
        'missing_statements': missing_statements,
        'missing_lines': missing_lines,
        'coverage_py_percent': coverage_py_percent,  # Coverage.py计算的百分比
        'branch_coverage': branch_coverage,
        'report_lines': report_lines,
        'results': results
    }


def print_detailed_results(result):
    """打印详细结果"""
    print(f"\\n{'='*70}")
    print(f"📊 {result['name'].upper()} 方法详细结果")
    print(f"{'='*70}")
    print(f"用例数: {result['test_cases']}")
    print(f"通过: {result['passed']} | 失败: {result['failed']}")
    print(f"通过率: {result['passed']/result['test_cases']*100:.1f}%")
    
    print(f"\\n📈 覆盖率统计:")
    print(f"  可执行语句: {result['total_statements']}")
    print(f"  已覆盖语句: {result['covered_statements']}")
    print(f"  未覆盖语句: {result['missing_statements']}")
    print(f"  语句覆盖率: {result['covered_statements']/result['total_statements']*100:.1f}%")
    if result['coverage_py_percent']:
        print(f"  Coverage.py综合覆盖率: {result['coverage_py_percent']:.1f}%")
    if result['branch_coverage']:
        print(f"  分支覆盖率: {result['branch_coverage']:.1f}%")
    
    print(f"\\n📋 Coverage.py 原始报告:")
    for line in result['report_lines']:
        print(line)
    
    print(f"\\n📝 用例执行详情 (前10个):")
    for r in result['results'][:10]:
        case = r['case']
        res = r['result']
        status = "✅" if res['passed'] else "❌"
        print(f"  {status} [{case['_case_type']}] {case['request']['method']} {case['request']['url']}")
        print(f"     预期: {res['expected']} | 实际: {res['status_code']} | 耗时: {res['elapsed']}s")
        if not res['passed'] and 'error' in res:
            print(f"     错误: {res['error']}")


def print_summary_table(results_list):
    """打印对比结果表格"""
    print("\\n" + "="*80)
    print("📈 对比实验结果汇总")
    print("="*80)
    print(f"{'方法':<10} {'用例数':<8} {'通过':<8} {'失败':<8} {'通过率':<10} {'语句覆盖':<10} {'综合覆盖':<10}")
    print("-"*80)
    
    for r in results_list:
        pass_rate = r['passed']/r['test_cases']*100 if r['test_cases'] > 0 else 0
        stmt_rate = r['covered_statements']/r['total_statements']*100 if r['total_statements'] > 0 else 0
        cov_rate = r['coverage_py_percent'] or stmt_rate
        
        print(f"{r['name']:<10} {r['test_cases']:<8} {r['passed']:<8} {r['failed']:<8} "
              f"{pass_rate:>6.1f}%    {stmt_rate:>6.1f}%    {cov_rate:>6.1f}%")
    
    print("="*80)
    
    # 找出最佳方法
    if results_list:
        best_coverage = max(results_list, key=lambda x: x['coverage_py_percent'] or 0)
        best_pass_rate = max(results_list, key=lambda x: x['passed']/x['test_cases'] if x['test_cases'] > 0 else 0)
        
        print(f"\\n🏆 最高综合覆盖率: {best_coverage['name']} ({best_coverage['coverage_py_percent']:.1f}%)")
        print(f"🏆 最高通过率: {best_pass_rate['name']} ({best_pass_rate['passed']}/{best_pass_rate['test_cases']})")


def main():
    """主对比实验"""
    print("="*80)
    print("🧪 API测试用例生成方法对比实验 (优化版)")
    print("="*80)
    
    api_desc = "POST /users 创建用户，参数：username(string,必填,3-20字符), age(integer,可选,0-120)"
    case_types = ['normal', 'boundary', 'exception']
    
    print(f"API描述: {api_desc}")
    print(f"用例类型: {case_types}")
    print(f"每种类型生成3个用例，共9个用例/方法")
    print(f"覆盖端点: POST /users, GET /users, GET /users/{{id}}, DELETE /users/{{id}}")
    
    results = []
    
    # 1. 随机生成
    result_random = measure_coverage('random', random_generate, api_desc, case_types)
    results.append(result_random)
    print_detailed_results(result_random)
    
    # 2. 规则生成
    result_rule = measure_coverage('rule', rule_generate, api_desc, case_types)
    results.append(result_rule)
    print_detailed_results(result_rule)
    
    # 3. 模型生成（如果可用）
    if MODEL_AVAILABLE:
        result_model = measure_coverage('model', model_generate, api_desc, case_types)
        results.append(result_model)
        print_detailed_results(result_model)
    else:
        print("\\n⚠️ 跳过模型生成对比（模型未加载）")
    
    # 打印汇总表格
    print_summary_table(results)
    
    # 保存详细结果到文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"comparison_result_{timestamp}.json"
    
    # 清理不可序列化的数据
    save_data = []
    for r in results:
        save_data.append({
            'name': r['name'],
            'test_cases': r['test_cases'],
            'passed': r['passed'],
            'failed': r['failed'],
            'pass_rate': r['passed']/r['test_cases']*100 if r['test_cases'] > 0 else 0,
            'total_statements': r['total_statements'],
            'covered_statements': r['covered_statements'],
            'statement_coverage': r['covered_statements']/r['total_statements']*100 if r['total_statements'] > 0 else 0,
            'coverage_py_percent': r['coverage_py_percent'],
            'branch_coverage': r['branch_coverage']
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    print(f"\\n💾 结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
'''

# 保存代码
with open('/D:\graduation_project\src/compare_v3.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("✅ 代码已保存到 /mnt/kimi/output/compare_v3.py")
print("\n主要优化点：")
print("1. 规则生成现在也覆盖 GET /users, GET /users/{id}, DELETE /users/{id}")
print("2. 添加了 calculate_expected_status() 函数，更准确地判断预期状态码")
print("3. 修正了覆盖率显示，使用 Coverage.py 原始报告的综合覆盖率")
print("4. 优化了用例分布统计，可以看到端点覆盖情况")
print("5. 改进了结果表格，显示通过率和两种覆盖率")
