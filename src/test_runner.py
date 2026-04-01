#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试执行器：将生成的测试用例发送到API，收集结果并生成HTML报告
支持超时、重试、请求间隔配置
"""

import json
import requests
import time
import os
from datetime import datetime
from typing import List, Dict, Any
import argparse

class TestRunner:
    def __init__(self, base_url: str = None, timeout: int = 10, retries: int = 0, delay: float = 0):
        """
        初始化测试执行器
        :param base_url: 可选的默认base URL，如果测试用例中url不完整则拼接
        :param timeout: 请求超时时间（秒）
        :param retries: 失败重试次数
        :param delay: 每次请求后的间隔时间（秒）
        """
        self.base_url = base_url
        self.timeout = timeout
        self.retries = retries
        self.delay = delay
        self.results = []
        self.session = requests.Session()
        # 设置超时
        self.session.timeout = (timeout, timeout)

    def execute_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行单个测试用例（支持重试）
        """
        request_info = test_case.get('request', {})
        expected = test_case.get('expected_response', {})

        method = request_info.get('method', 'GET').upper()
        url = request_info.get('url', '')
        headers = request_info.get('headers', {})
        params = request_info.get('query', {})
        body = request_info.get('body', None)

        if self.base_url and not url.startswith(('http://', 'https://')):
            url = self.base_url.rstrip('/') + '/' + url.lstrip('/')

        req_kwargs = {
            'method': method,
            'url': url,
            'headers': headers,
            'params': params,
            'timeout': self.timeout
        }
        if method in ['POST', 'PUT', 'PATCH'] and body:
            if headers.get('Content-Type') == 'application/json':
                req_kwargs['json'] = body
            else:
                req_kwargs['data'] = body

        # 重试逻辑
        for attempt in range(self.retries + 1):
            start_time = time.time()
            error = None
            response = None
            try:
                response = self.session.request(**req_kwargs)
                status_code = response.status_code
                response_body = response.text[:500]
                break
            except requests.exceptions.Timeout:
                error = "Timeout"
                status_code = None
                response_body = None
            except requests.exceptions.ConnectionError:
                error = "Connection Error"
                status_code = None
                response_body = None
            except Exception as e:
                error = str(e)
                status_code = None
                response_body = None
            if attempt < self.retries:
                time.sleep(self.delay)  # 重试前等待
        else:
            # 所有重试都失败
            status_code = None
            error = error or "Max retries exceeded"

        elapsed = time.time() - start_time
        expected_status = expected.get('status_code', 200)
        passed = (status_code == expected_status) if status_code is not None else False

        result = {
            'test_case': test_case,
            'timestamp': datetime.now().isoformat(),
            'request': {
                'method': method,
                'url': url,
                'headers': headers,
                'params': params,
                'body': body
            },
            'response': {
                'status_code': status_code,
                'elapsed': elapsed,
                'body_preview': response_body,
                'error': error
            },
            'expected_status': expected_status,
            'passed': passed
        }

        # 请求间隔
        time.sleep(self.delay)
        return result

    def run(self, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量执行测试用例"""
        self.results = []
        total = len(test_cases)
        for i, case in enumerate(test_cases, 1):
            print(f"执行测试 {i}/{total}: {case.get('description', '')[:50]}...")
            result = self.execute_test_case(case)
            self.results.append(result)
            status = "✅" if result['passed'] else "❌"
            print(f"   {status} 状态码: {result['response']['status_code']}, 耗时: {result['response']['elapsed']:.2f}s")
        return self.results


    def generate_html_report(self, output_path: str = "reports/test_report.html"):
        """
        生成HTML测试报告
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 统计数据
        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        failed = total - passed
        
        # 生成HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>API测试报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .detail {{ max-width: 300px; overflow: auto; }}
        .status-pass {{ background-color: #d4edda; }}
        .status-fail {{ background-color: #f8d7da; }}
    </style>
</head>
<body>
    <h1>API自动化测试报告</h1>
    <div class="summary">
        <p><strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>总用例数:</strong> {total}</p>
        <p><strong>通过:</strong> <span class="pass">{passed}</span> ({passed/total*100:.1f}%)</p>
        <p><strong>失败:</strong> <span class="fail">{failed}</span> ({failed/total*100:.1f}%)</p>
    </div>
    
    <table>
        <tr>
            <th>序号</th>
            <th>描述</th>
            <th>方法</th>
            <th>URL</th>
            <th>期望状态</th>
            <th>实际状态</th>
            <th>耗时(秒)</th>
            <th>状态</th>
            <th>详情</th>
        </tr>
        """
        
        for idx, r in enumerate(self.results, 1):
            desc = r['test_case'].get('description', 'N/A')
            method = r['request']['method']
            url = r['request']['url']
            exp = r['expected_status']
            act = r['response']['status_code'] if r['response']['status_code'] else 'ERROR'
            elapsed = f"{r['response']['elapsed']:.3f}"
            status_class = "status-pass" if r['passed'] else "status-fail"
            status_icon = "✅" if r['passed'] else "❌"
            # 详情预览
            if r['response']['error']:
                detail = f"错误: {r['response']['error']}"
            else:
                detail = f"响应体: {r['response']['body_preview'][:100]}"
            
            html += f"""
        <tr class="{status_class}">
            <td>{idx}</td>
            <td>{desc}</td>
            <td>{method}</td>
            <td>{url}</td>
            <td>{exp}</td>
            <td>{act}</td>
            <td>{elapsed}</td>
            <td>{status_icon}</td>
            <td class="detail">{detail}</td>
        </tr>"""
        
        html += """
    </table>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✅ 报告已生成: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="执行API测试用例并生成报告")
    parser.add_argument('--cases', type=str, help='测试用例JSON文件路径')
    parser.add_argument('--base-url', type=str, default='http://httpbin.org', help='API基础URL')
    parser.add_argument('--output', type=str, default='reports/test_report.html', help='报告输出路径')
    args = parser.parse_args()
    
    # 如果没有提供用例文件，则使用一个简单的示例用例列表（可替换为从生成器获取）
    if args.cases:
        with open(args.cases, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    else:
        # 示例用例（可直接用之前生成的用例）
        print("未指定用例文件，使用内置示例用例（指向httpbin.org）")
        test_cases = [
            {
                "description": "GET 测试 - 成功",
                "request": {
                    "method": "GET",
                    "url": "/get",
                    "headers": {},
                    "query": {"foo": "bar"}
                },
                "expected_response": {"status_code": 200}
            },
            {
                "description": "POST 测试 - 创建数据",
                "request": {
                    "method": "POST",
                    "url": "/post",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"name": "test", "value": 123}
                },
                "expected_response": {"status_code": 200}
            },
            {
                "description": "404 测试 - 期望失败",
                "request": {
                    "method": "GET",
                    "url": "/status/404",
                    "headers": {}
                },
                "expected_response": {"status_code": 404}
            }
        ]
    
    runner = TestRunner(base_url=args.base_url)
    results = runner.run(test_cases)
    report_path = runner.generate_html_report(args.output)
    print(f"\n测试完成！报告已保存至: {report_path}")

if __name__ == "__main__":
    main()