#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API测试用例生成器 - 模型推理脚本
修复版：支持三类用例生成、格式统一、批量处理、与规则引擎结合
"""

import json
import torch
import os
import sys
import re
from typing import List, Dict, Union, Optional
from datetime import datetime

# 尝试导入规则引擎（如果有）
try:
    sys.path.append('src')
    from generate_samples_from_openapi import ValueGenerator, TestCaseBuilder
    RULE_ENGINE_AVAILABLE = True
except ImportError:
    RULE_ENGINE_AVAILABLE = False
    print("⚠️ 规则引擎未找到，将使用纯模型生成")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ========== 配置 ==========

MODEL_PATH = os.path.abspath("models/fine_tuned_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成参数配置
GENERATION_CONFIG = {
    'normal': {
        'num_beams': 3,
        'temperature': 1.0,
        'do_sample': False,
        'max_length': 128,
        'description': '正常值用例'
    },
    'boundary': {
        'num_beams': 3,
        'temperature': 0.8,  # 稍低，更确定
        'do_sample': False,
        'max_length': 128,
        'description': '边界值用例'
    },
    'exception': {
        'num_beams': 3,
        'temperature': 1.2,  # 稍高，更多样
        'do_sample': True,   # 采样增加多样性
        'top_p': 0.9,
        'max_length': 128,
        'description': '异常值用例'
    }
}


# ========== 模型加载 ==========

def load_model(model_path: str = MODEL_PATH, fallback_to_base: bool = True):
    """
    加载模型，支持本地微调模型和基础模型回退
    
    Args:
        model_path: 模型路径
        fallback_to_base: 如果本地模型不存在，是否加载基础T5模型
    
    Returns:
        (tokenizer, model, model_source)
    """
    print(f"🔍 查找模型: {model_path}")
    
    # 检查本地模型
    if os.path.exists(model_path):
        required_files = ['config.json', 'pytorch_model.bin']
        missing = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
        
        if not missing:
            print(f"✅ 找到本地微调模型")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True).to(DEVICE)
                model.eval()
                return tokenizer, model, "local_finetuned"
            except Exception as e:
                print(f"⚠️ 加载本地模型失败: {e}")
                if not fallback_to_base:
                    raise
    
    # 回退到基础模型
    if fallback_to_base:
        base_model = "t5-small"  # 或你训练用的基础模型
        print(f"⚠️ 本地模型不可用，回退到基础模型: {base_model}")
        print("   生成效果可能较差，建议先完成训练")
        
        try:
            # 设置镜像源
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            model = AutoModelForSeq2SeqLM.from_pretrained(base_model).to(DEVICE)
            model.eval()
            return tokenizer, model, "base_model"
        except Exception as e:
            print(f"❌ 加载基础模型也失败: {e}")
            raise
    
    raise FileNotFoundError(f"模型未找到: {model_path}")


# ========== 输入格式化 ==========

def format_api_description(method: str, path: str, 
                        parameters: List[Dict] = None,
                        description: str = "") -> str:
    """
    统一格式化API描述，确保与训练时格式一致
    
    Args:
        method: HTTP方法 (GET/POST/PUT/DELETE)
        path: API路径
        parameters: 参数列表，每个参数是dict
        description: 额外描述
    
    Returns:
        格式化后的描述字符串
    """
    parts = [f"{method.upper()} {path}"]
    
    if description:
        parts.append(f"- {description}")
    
    if parameters:
        param_strs = []
        for p in parameters:
            name = p.get('name', 'unknown')
            ptype = p.get('type', 'string')
            required = "必填" if p.get('required') else "可选"
            
            # 约束信息
            constraints = []
            if 'minimum' in p:
                constraints.append(f"最小值={p['minimum']}")
            if 'maximum' in p:
                constraints.append(f"最大值={p['maximum']}")
            if 'minLength' in p:
                constraints.append(f"最小长度={p['minLength']}")
            if 'maxLength' in p:
                constraints.append(f"最大长度={p['maxLength']}")
            if 'enum' in p:
                constraints.append(f"枚举={p['enum']}")
            
            cons_str = f", {', '.join(constraints)}" if constraints else ""
            param_strs.append(f"{name}({ptype},{required}{cons_str})")
        
        parts.append(f"参数: {', '.join(param_strs)}")
    
    return " ".join(parts)


def parse_api_description(desc: str) -> Dict:
    """
    反向解析API描述（简单版本）
    用于处理自由文本输入
    
    Args:
        desc: 自由格式描述
    
    Returns:
        结构化dict
    """
    # 尝试匹配 METHOD PATH 格式
    match = re.match(r'(GET|POST|PUT|DELETE|PATCH)\s+(\S+)(.*)', desc, re.IGNORECASE)
    if match:
        method = match.group(1).upper()
        path = match.group(2)
        rest = match.group(3).strip()
        
        # 简单提取参数信息
        params = []
        # 匹配 参数名(类型,...) 格式
        param_pattern = r'(\w+)\s*\(\s*(\w+)\s*,\s*(必填|可选)([^)]*)\)'
        for m in re.finditer(param_pattern, rest):
            param = {
                'name': m.group(1),
                'type': m.group(2),
                'required': m.group(3) == '必填'
            }
            # 解析约束
            cons = m.group(4)
            if '最小值=' in cons:
                min_match = re.search(r'最小值=(-?\d+)', cons)
                if min_match:
                    param['minimum'] = int(min_match.group(1))
            if '最大值=' in cons:
                max_match = re.search(r'最大值=(-?\d+)', cons)
                if max_match:
                    param['maximum'] = int(max_match.group(1))
            
            params.append(param)
        
        return {
            'method': method,
            'path': path,
            'parameters': params,
            'description': rest
        }
    
    # 无法解析，返回原始
    return {'raw': desc}


# ========== 核心生成函数 ==========

def generate_with_model(api_description: str, tokenizer, model, 
                       case_type: str = 'normal') -> Dict:
    """
    使用模型生成测试用例
    
    Args:
        api_description: API描述文本
        tokenizer: tokenizer
        model: 模型
        case_type: 用例类型 (normal/boundary/exception)
    
    Returns:
        生成的测试用例dict
    """
    config = GENERATION_CONFIG.get(case_type, GENERATION_CONFIG['normal'])
    
    # 添加类型提示（帮助模型区分）
    prompt = f"[{case_type.upper()}] {api_description}"
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)
    
    # 生成参数
    gen_kwargs = {
        'max_length': config['max_length'],
        'num_beams': config['num_beams'],
        'early_stopping': True,
        'no_repeat_ngram_size': 2,
        'do_sample': config.get('do_sample', False),
    }
    
    if config.get('do_sample'):
        gen_kwargs['temperature'] = config.get('temperature', 1.0)
        if 'top_p' in config:
            gen_kwargs['top_p'] = config['top_p']
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    # 解码
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 尝试解析JSON
    return parse_generated_output(decoded)


def parse_generated_output(decoded: str) -> Dict:
    """
    解析模型生成的输出，处理各种格式问题
    
    Args:
        decoded: 模型生成的原始字符串
    
    Returns:
        解析后的dict
    """
    decoded = decoded.strip()
    
    # 尝试直接解析
    try:
        return json.loads(decoded)
    except json.JSONDecodeError:
        pass
    
    # 尝试提取JSON块
    # 匹配 {...} 或 [...]
    patterns = [
        (r'\{[^{}]*\}', 'object'),  # 简单对象
        (r'\{[^{}]*\{[^{}]*\}[^{}]*\}', 'nested_object'),  # 嵌套一层
        (r'\[[^\[\]]*\]', 'array'),  # 数组
    ]
    
    for pattern, ptype in patterns:
        matches = re.findall(pattern, decoded, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except:
                continue
    
    # 尝试清理常见污染
    cleaned = decoded
    # 移除markdown代码块标记
    cleaned = re.sub(r'```json\s*', '', cleaned)
    cleaned = re.sub(r'```\s*', '', cleaned)
    # 移除解释性文字
    cleaned = re.sub(r'^[^{[]*', '', cleaned)
    cleaned = re.sub(r'[^}\]]*$', '', cleaned)
    
    try:
        return json.loads(cleaned)
    except:
        pass
    
    # 最终fallback：返回原始文本
    return {
        "parse_error": True,
        "raw_output": decoded,
        "note": "模型输出无法解析为JSON，请检查训练数据格式"
    }


def generate_hybrid(api_description: str, tokenizer, model,
                   case_type: str = 'normal',
                   use_rule_fallback: bool = True) -> Dict:
    """
    混合生成：先用模型，失败时用规则回退
    
    Args:
        api_description: API描述
        tokenizer: tokenizer
        model: 模型
        case_type: 用例类型
        use_rule_fallback: 是否使用规则引擎回退
    
    Returns:
        测试用例dict
    """
    # 尝试模型生成
    result = generate_with_model(api_description, tokenizer, model, case_type)
    
    # 检查是否成功
    if 'parse_error' not in result and 'error' not in result:
        result['_generation_method'] = 'model'
        result['_case_type'] = case_type
        return result
    
    # 模型失败，尝试规则回退
    if use_rule_fallback and RULE_ENGINE_AVAILABLE:
        print(f"   ⚠️ 模型生成失败，使用规则引擎回退")
        try:
            # 解析描述
            parsed = parse_api_description(api_description)
            if 'method' in parsed:
                # 构造endpoint_info格式
                endpoint_info = {
                    'method': parsed['method'],
                    'path': parsed['path'],
                    'parameters': parsed.get('parameters', []),
                    'requestBody': None
                }
                # 使用规则生成
                builder = TestCaseBuilder()
                rule_result = builder.build(endpoint_info, case_type)
                rule_result['_generation_method'] = 'rule_fallback'
                rule_result['_case_type'] = case_type
                return rule_result
        except Exception as e:
            print(f"   ❌ 规则回退也失败: {e}")
    
    # 都失败了，返回错误信息
    result['_generation_method'] = 'failed'
    result['_case_type'] = case_type
    return result


# ========== 批量生成 ==========

def generate_all_case_types(api_description: str, tokenizer, model,
                           use_hybrid: bool = True) -> Dict[str, Dict]:
    """
    为单个API生成三类测试用例
    
    Args:
        api_description: API描述
        tokenizer: tokenizer
        model: 模型
        use_hybrid: 是否使用混合生成
    
    Returns:
        {'normal': {...}, 'boundary': {...}, 'exception': {...}}
    """
    results = {}
    
    print(f"\n📝 API: {api_description[:60]}...")
    
    for case_type in ['normal', 'boundary', 'exception']:
        print(f"   生成 [{case_type}]...", end=" ")
        
        if use_hybrid:
            result = generate_hybrid(api_description, tokenizer, model, case_type)
        else:
            result = generate_with_model(api_description, tokenizer, model, case_type)
        
        # 检查成功
        if 'parse_error' in result or 'error' in result:
            print("❌ 失败")
        else:
            print("✅ 成功")
        
        results[case_type] = result
    
    return results


def batch_generate(descriptions: List[str], tokenizer, model,
                   output_file: str = None) -> List[Dict]:
    """
    批量生成测试用例
    
    Args:
        descriptions: API描述列表
        tokenizer: tokenizer
        model: 模型
        output_file: 可选，保存结果到文件
    
    Returns:
        结果列表
    """
    all_results = []
    total = len(descriptions)
    
    print(f"\n🚀 批量生成 {total} 个API的测试用例")
    print("=" * 60)
    
    for i, desc in enumerate(descriptions, 1):
        print(f"\n[{i}/{total}]")
        results = generate_all_case_types(desc, tokenizer, model)
        
        all_results.append({
            'api_description': desc,
            'test_cases': results,
            'timestamp': datetime.now().isoformat()
        })
        
        # 进度显示
        if i % 5 == 0:
            print(f"   进度: {i}/{total} ({i/total*100:.1f}%)")
    
    # 保存结果
    if output_file:
        output_data = {
            'generated_at': datetime.now().isoformat(),
            'model_source': 'local' if os.path.exists(MODEL_PATH) else 'base',
            'total_apis': total,
            'results': all_results
        }
        
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 结果已保存: {output_file}")
    
    return all_results


# ========== 主函数 ==========

def main():
    print("=" * 60)
    print("API测试用例生成器 - 模型推理")
    print("=" * 60)
    
    # 加载模型
    try:
        tokenizer, model, model_source = load_model()
        print(f"✅ 模型加载成功 (来源: {model_source})")
        print(f"   设备: {DEVICE}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)
    
    # 准备测试输入（使用统一格式）
    test_inputs = [
        # 方式1：结构化输入（推荐）
        format_api_description(
            "POST", "/api/users",
            parameters=[
                {'name': 'username', 'type': 'string', 'required': True, 'minLength': 3, 'maxLength': 20},
                {'name': 'age', 'type': 'integer', 'required': False, 'minimum': 0, 'maximum': 120},
                {'name': 'email', 'type': 'string', 'required': True, 'format': 'email'}
            ],
            description="创建新用户"
        ),
        
        # 方式2：自由文本（会尝试解析）
        "GET /api/pet/{petId} 参数: petId(integer,必填)",
        
        # 方式3：简单描述
        "DELETE /api/users/{id} 根据ID删除用户，id为整数",
    ]
    
    print(f"\n📝 准备生成 {len(test_inputs)} 个API的测试用例")
    
    # 单个生成演示
    print("\n" + "=" * 60)
    print("单API生成演示")
    print("=" * 60)
    
    for desc in test_inputs[:2]:  # 前两个做详细展示
        results = generate_all_case_types(desc, tokenizer, model)
        
        print(f"\n📋 API描述: {desc}")
        for case_type, result in results.items():
            print(f"\n   [{case_type}]:")
            print(f"   方法: {result.get('_generation_method', 'unknown')}")
            # 美化输出
            result_clean = {k: v for k, v in result.items() if not k.startswith('_')}
            print(f"   内容: {json.dumps(result_clean, indent=6, ensure_ascii=False)}")
    
    # 批量生成
    print("\n" + "=" * 60)
    print("批量生成")
    print("=" * 60)
    
    batch_results = batch_generate(
        test_inputs, 
        tokenizer, model,
        output_file="output/generated_test_cases.json"
    )
    
    # 统计
    print("\n" + "=" * 60)
    print("生成统计")
    print("=" * 60)
    
    total_success = 0
    total_failed = 0
    
    for api_result in batch_results:
        for case_type, tc in api_result['test_cases'].items():
            if 'parse_error' in tc or 'error' in tc:
                total_failed += 1
            else:
                total_success += 1
    
    total = total_success + total_failed
    print(f"总计生成: {total} 个用例")
    print(f"  ✅ 成功: {total_success} ({total_success/total*100:.1f}%)")
    print(f"  ❌ 失败: {total_failed} ({total_failed/total*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()