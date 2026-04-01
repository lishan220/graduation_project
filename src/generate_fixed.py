#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API测试用例生成器 - 修复版
修复：模型加载超时、增加纯规则回退、离线支持
"""

import json
import torch
import os
import sys
import re
from typing import Dict, List
from datetime import datetime

# 设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.abspath("models/fine_tuned_model")

# 尝试导入transformers
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  transformers未安装，将使用纯规则生成")
    TRANSFORMERS_AVAILABLE = False


# ========== 纯规则生成器（不依赖模型）==========

class RuleBasedGenerator:
    """纯规则生成器，作为模型加载失败的回退"""
    
    def generate(self, api_desc: str, case_type: str = 'normal') -> Dict:
        """基于规则生成测试用例"""
        
        # 解析API描述
        parsed = self._parse_description(api_desc)
        
        method = parsed.get('method', 'GET')
        path = parsed.get('path', '/api/unknown')
        params = parsed.get('params', [])
        
        # 构建请求
        request = {
            "method": method,
            "url": f"http://api.example.com{path}",
            "headers": {},
            "query": {},
            "body": None
        }
        
        # 根据类型生成参数值
        for param in params:
            name = param.get('name', 'param')
            ptype = param.get('type', 'string')
            
            value = self._generate_value(ptype, case_type, param)
            
            # 决定放query还是body
            if method in ['GET', 'DELETE']:
                request['query'][name] = value
            else:
                if request['body'] is None:
                    request['body'] = {}
                request['body'][name] = value
        
        return {
            "description": f"{method} {path} - {case_type} 用例（规则生成）",
            "request": request,
            "expected_response": {
                "status_code": 200 if case_type == 'normal' else (400 if case_type == 'exception' else 200)
            },
            "_generation_method": "rule_fallback",
            "_case_type": case_type
        }
    
    def _parse_description(self, desc: str) -> Dict:
        """简单解析API描述"""
        result = {'method': 'GET', 'path': '/', 'params': []}
        
        # 匹配 METHOD PATH
        match = re.match(r'(GET|POST|PUT|DELETE)\s+(\S+)', desc, re.IGNORECASE)
        if match:
            result['method'] = match.group(1).upper()
            result['path'] = match.group(2)
        
        # 简单提取参数（简化版）
        if '参数' in desc or 'params' in desc.lower():
            # 提取 name(type) 模式
            param_pattern = r'(\w+)\s*[\(（](\w+)[\)）]'
            for m in re.finditer(param_pattern, desc):
                result['params'].append({
                    'name': m.group(1),
                    'type': m.group(2),
                    'required': '必填' in desc or 'required' in desc.lower()
                })
        
        return result
    
    def _generate_value(self, ptype: str, case_type: str, param_info: Dict):
        """生成参数值"""
        
        if ptype == 'integer':
            if case_type == 'normal':
                return 25
            elif case_type == 'boundary':
                return 0
            else:  # exception
                return -1
        
        elif ptype == 'string':
            if case_type == 'normal':
                return "test_string"
            elif case_type == 'boundary':
                return ""
            else:
                return "'; DROP TABLE users; --"
        
        elif ptype == 'boolean':
            return True if case_type == 'normal' else "not_bool"
        
        else:
            return "unknown"


# ========== 模型生成器 ==========

class ModelBasedGenerator:
    """基于微调模型的生成器"""
    
    def __init__(self, model_path: str):
        self.tokenizer = None
        self.model = None
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """加载模型，带多重回退"""
        
        # 1. 尝试本地微调模型
        if os.path.exists(self.model_path):
            try:
                print(f"   加载本地模型: {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, 
                    local_files_only=True
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_path,
                    local_files_only=True
                ).to(DEVICE)
                self.model.eval()
                print("   ✅ 本地模型加载成功")
                return
            except Exception as e:
                print(f"   ⚠️  本地模型加载失败: {e}")
        
        # 2. 尝试基础T5模型（预下载的）
        base_model = "t5-small"
        try:
            print(f"   加载基础模型: {base_model}")
            # 先尝试本地缓存
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                local_files_only=True
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model,
                local_files_only=True
            ).to(DEVICE)
            self.model.eval()
            print("   ⚠️  使用基础模型（未微调），效果可能较差")
            return
        except Exception as e:
            print(f"   ❌ 基础模型也失败: {e}")
            raise RuntimeError("模型加载失败，将使用规则引擎")
    
    def generate(self, api_desc: str, case_type: str = 'normal') -> Dict:
        """使用模型生成"""
        
        # 添加类型标记
        prompt = f"[{case_type.upper()}] {api_desc}"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(DEVICE)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=3,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        # 解码
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 尝试解析JSON
        try:
            result = json.loads(decoded)
            result['_generation_method'] = 'model'
            result['_case_type'] = case_type
            return result
        except json.JSONDecodeError:
            # 尝试提取JSON
            match = re.search(r'\{.*\}', decoded, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                    result['_generation_method'] = 'model_extracted'
                    result['_case_type'] = case_type
                    return result
                except:
                    pass
            
            # 失败，返回原始
            return {
                "parse_error": True,
                "raw_output": decoded,
                "_generation_method": "model_failed",
                "_case_type": case_type
            }


# ========== 主生成器（智能选择）==========

class TestCaseGenerator:
    """智能生成器：优先模型，失败则规则"""
    
    def __init__(self):
        self.model_gen = None
        self.rule_gen = RuleBasedGenerator()
        
        # 尝试加载模型
        if TRANSFORMERS_AVAILABLE:
            try:
                self.model_gen = ModelBasedGenerator(MODEL_PATH)
            except Exception as e:
                print(f"⚠️  模型加载失败，将使用纯规则生成: {e}")
    
    def generate(self, api_desc: str, case_type: str = 'normal') -> Dict:
        """生成单个用例"""
        
        # 优先使用模型
        if self.model_gen:
            try:
                result = self.model_gen.generate(api_desc, case_type)
                # 检查是否成功解析
                if 'parse_error' not in result:
                    return result
                # 解析失败，降级到规则
                print(f"   [{case_type}] 模型解析失败，使用规则")
            except Exception as e:
                print(f"   [{case_type}] 模型生成错误: {e}")
        
        # 使用规则
        return self.rule_gen.generate(api_desc, case_type)
    
    def generate_all_types(self, api_desc: str) -> Dict[str, Dict]:
        """生成三类用例"""
        return {
            'normal': self.generate(api_desc, 'normal'),
            'boundary': self.generate(api_desc, 'boundary'),
            'exception': self.generate(api_desc, 'exception')
        }


# ========== 工具函数 ==========

def format_api_description(method: str, path: str, params: List[Dict] = None) -> str:
    """格式化API描述"""
    parts = [f"{method.upper()} {path}"]
    
    if params:
        param_strs = []
        for p in params:
            name = p.get('name', 'unknown')
            ptype = p.get('type', 'string')
            required = "必填" if p.get('required') else "可选"
            param_strs.append(f"{name}({ptype},{required})")
        parts.append(f"参数: {', '.join(param_strs)}")
    
    return " ".join(parts)


# ========== 主函数 ==========

def main():
    print("=" * 60)
    print("API测试用例生成器")
    print("=" * 60)
    
    # 初始化生成器
    print("\n🔧 初始化生成器...")
    generator = TestCaseGenerator()
    
    # 测试用例
    test_apis = [
        # 格式1：结构化
        format_api_description("POST", "/api/users", [
            {'name': 'username', 'type': 'string', 'required': True},
            {'name': 'age', 'type': 'integer', 'required': False}
        ]),
        # 格式2：简单描述
        "GET /api/pet/{petId} 参数: petId(integer,必填)",
        # 格式3：更自由的描述
        "DELETE /api/users/{id} 根据ID删除用户",
    ]
    
    print(f"\n📝 生成 {len(test_apis)} 个API的测试用例")
    
    all_results = []
    
    for i, api_desc in enumerate(test_apis, 1):
        print(f"\n[{i}/{len(test_apis)}] {api_desc[:50]}...")
        
        # 生成三类
        results = generator.generate_all_types(api_desc)
        
        for case_type, result in results.items():
            method = result.get('_generation_method', 'unknown')
            status = "✅" if 'parse_error' not in result and 'error' not in result else "⚠️"
            print(f"   {status} [{case_type}] ({method})")
        
        all_results.append({
            'api': api_desc,
            'test_cases': results
        })
    
    # 保存结果
    output_file = "output/generated_results.json"
    os.makedirs("output", exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'results': all_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存: {output_file}")
    
    # 显示示例
    print("\n" + "=" * 60)
    print("生成示例（第一个API的正常值）:")
    print("=" * 60)
    example = all_results[0]['test_cases']['normal']
    print(json.dumps(example, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 60)
    print("🎉 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()