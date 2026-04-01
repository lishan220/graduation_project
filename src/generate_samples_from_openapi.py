import json
import os
import sys
import random
from datetime import datetime

# 添加src路径以便导入api_parser
sys.path.append('src')
from api_parser import OpenAPIParser

def generate_value_by_type(param_info, case_type='normal'):
    """
    根据参数类型和约束生成一个测试值
    param_info: 参数信息字典（包含type, constraints等）
    case_type: normal/boundary/exception
    """
    param_type = param_info.get('type', 'string')
    constraints = param_info.get('constraints', {})

    # 处理枚举值
    if constraints.get('enum'):
        if case_type == 'normal':
            return random.choice(constraints['enum'])
        elif case_type == 'exception':
            # 生成一个不在枚举中的值
            if param_type == 'string':
                return "invalid_enum_value"
            else:
                return 9999

    # 根据类型生成
    if param_type == 'integer':
        minimum = constraints.get('minimum')
        maximum = constraints.get('maximum')

        if case_type == 'normal':
            if minimum is not None and maximum is not None:
                return random.randint(minimum, maximum)
            elif minimum is not None:
                return minimum + 5
            elif maximum is not None:
                return max(1, maximum - 5)
            else:
                return random.randint(1, 100)
        elif case_type == 'boundary':
            if minimum is not None and maximum is not None:
                return random.choice([minimum, maximum])
            elif minimum is not None:
                return minimum
            elif maximum is not None:
                return maximum
            else:
                return 0
        elif case_type == 'exception':
            if minimum is not None:
                return minimum - 1
            elif maximum is not None:
                return maximum + 1
            else:
                return -999

    elif param_type == 'string':
        min_len = constraints.get('minLength', 1)
        if min_len is None:          # 如果 minLength 字段存在但值为 None，也设置为默认值
            min_len = 1
        max_len = constraints.get('maxLength')
        if max_len is None:
            max_len = 50              # 如果没有定义最大长度，设定默认最大值 50
        if case_type == 'normal':
            length = random.randint(min_len, max_len)
            return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))
        elif case_type == 'boundary':
            # 边界值：最小长度和最大长度
            if random.choice([True, False]):
                return 'a' * min_len
            else:
                return 'b' * max_len
        elif case_type == 'exception':
            # 异常值：超出最大长度或为空
            if random.choice([True, False]):
                return ''  # 空字符串
            else:
                return 'c' * (max_len + 5)

    elif param_type == 'boolean':
        if case_type == 'normal':
            return random.choice([True, False])
        elif case_type == 'boundary':
            return True  # 布尔值边界就是两个值
        else:
            return "not_a_boolean"  # 异常值

    else:
        # 默认返回字符串
        return "test_value"

def build_request(endpoint_info, case_type='normal'):
    """
    根据端点信息和用例类型构建一个完整的请求字典
    """
    method = endpoint_info['method']
    path = endpoint_info['path']
    
    # 初始化请求结构
    request = {
        "method": method,
        "url": f"http://api.example.com{path}",  # 可配置base_url
        "headers": {},
        "body": {},
        "query": {}
    }
    
    # 处理路径参数（需要替换URL中的占位符）
    path_params = {}
    for param in endpoint_info['parameters']:
        if param['in'] == 'path':
            value = generate_value_by_type(param, case_type)
            path_params[param['name']] = value
    
    # 替换URL中的路径参数
    for name, value in path_params.items():
        placeholder = f"{{{name}}}"
        if placeholder in request['url']:
            request['url'] = request['url'].replace(placeholder, str(value))
        else:
            # 如果没有占位符，添加到query？这里简单处理
            pass
    
    # 处理query参数
    for param in endpoint_info['parameters']:
        if param['in'] == 'query':
            # 对于正常和边界，只包含必要的参数；异常时可以包含额外参数
            if case_type == 'normal' and not param['required']:
                # 可选参数随机决定是否包含
                if random.random() > 0.5:
                    value = generate_value_by_type(param, case_type)
                    request['query'][param['name']] = value
            elif case_type == 'normal' and param['required']:
                value = generate_value_by_type(param, case_type)
                request['query'][param['name']] = value
            elif case_type == 'boundary':
                # 边界值：包含所有参数，并生成边界值
                value = generate_value_by_type(param, 'boundary')
                request['query'][param['name']] = value
            elif case_type == 'exception':
                # 异常值：可以缺失必填参数，或生成异常值
                if param['required'] and random.random() > 0.3:
                    # 缺失必填参数（不包含）
                    pass
                else:
                    value = generate_value_by_type(param, 'exception')
                    request['query'][param['name']] = value
    
    # 处理请求体（如果有）
    if endpoint_info['requestBody'] and endpoint_info['requestBody']['schema']:
        schema = endpoint_info['requestBody']['schema']
        if schema.get('type') == 'object':
            body_params = {}
            properties = schema.get('properties', {})
            required_list = schema.get('required', [])
            
            for prop_name, prop_schema in properties.items():
                # 构建一个临时的参数信息
                param_info = {
                    'type': prop_schema.get('type'),
                    'constraints': {
                        'minimum': prop_schema.get('minimum'),
                        'maximum': prop_schema.get('maximum'),
                        'minLength': prop_schema.get('minLength'),
                        'maxLength': prop_schema.get('maxLength'),
                        'enum': prop_schema.get('enum')
                    }
                }
                
                # 根据用例类型决定是否包含该字段
                if case_type == 'normal':
                    if prop_name in required_list or random.random() > 0.3:
                        value = generate_value_by_type(param_info, 'normal')
                        body_params[prop_name] = value
                elif case_type == 'boundary':
                    # 边界值：所有字段都包含，并用边界值
                    value = generate_value_by_type(param_info, 'boundary')
                    body_params[prop_name] = value
                elif case_type == 'exception':
                    # 异常值：可以缺失必填字段，或用异常值
                    if prop_name in required_list and random.random() > 0.5:
                        # 缺失必填字段
                        pass
                    else:
                        value = generate_value_by_type(param_info, 'exception')
                        body_params[prop_name] = value
            
            if body_params:
                request['body'] = body_params
                request['headers']['Content-Type'] = 'application/json'
    
    return request

def main():
    # ===== 配置区域 =====
    # 指定要处理的API文档文件列表（可以根据需要增删）
    api_files = [
        "data/raw/api_docs/petstore.json",
        "data/raw/api_docs/Events API-swagger.json",
        "data/raw/api_docs/Platform API-swagger.json",
        "data/raw/api_docs/1Forge Finance APIs-swagger.json",
        "data/raw/api_docs/1Password Connect-swagger.json",
        "data/raw/api_docs/Adyen Test Cards API-swagger.json",
        "data/raw/api_docs/Authentiq API-swagger.json",
        "data/raw/api_docs/Control API v1-swagger.json",
        "data/raw/api_docs/Hosted onboarding API-swagger.json",
        "data/raw/api_docs/IP geolocation API-swagger.json",
        "data/raw/api_docs/Management API-swagger.json",
        "data/raw/api_docs/Adyen Balance Control API-swagger.json",
        # 可以继续添加其他文件
    ]

    # 每个端点每种类型生成的样本数
    SAMPLES_PER_TYPE = 5  # 可以根据需要调整，例如 8 或 10

    # ===== 处理过程 =====
    all_dataset = []
    parser = OpenAPIParser()
    case_types = ['normal', 'boundary', 'exception']

    for file_path in api_files:
        if not os.path.exists(file_path):
            print(f"⚠️ 文件不存在，跳过: {file_path}")
            continue

        print(f"\n📄 处理文件: {file_path}")
        try:
            parser.load(file_path)
            endpoints = parser.extract_endpoints()
            print(f"   解析到 {len(endpoints)} 个端点")
        except Exception as e:
            print(f"   解析失败: {e}")
            continue

        for idx, ep in enumerate(endpoints):
            api_desc = parser.generate_description(ep)
            for case_type in case_types:
                for sample_idx in range(SAMPLES_PER_TYPE):
                    request = build_request(ep, case_type)
                    test_case = {
                        "description": f"{ep['method']} {ep['path']} - {case_type} 用例 {sample_idx+1}",
                        "request": request,
                        "expected_response": {"status_code": 200 if case_type != 'exception' else 400}
                    }
                    sample = {
                        "id": f"auto_{os.path.basename(file_path).split('.')[0]}_{idx:03d}_{case_type}_{sample_idx}",
                        "api_description": api_desc,
                        "endpoint": ep['path'],
                        "method": ep['method'],
                        "case_type": case_type,
                        "test_case": test_case,
                        "created_at": datetime.now().isoformat()
                    }
                    all_dataset.append(sample)

    # 保存合并后的数据集
    output_path = "data/processed/auto_dataset.json"
    os.makedirs("data/processed", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 总共生成 {len(all_dataset)} 个测试用例，保存到 {output_path}")

if __name__ == "__main__":
    main()