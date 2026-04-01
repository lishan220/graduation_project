#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合生成器 v2 - 支持三类用例生成（正常/边界/异常）
"""

import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "models/fine_tuned_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RuleBasedGenerator:
    """基于规则的测试用例生成器 - 支持三类用例"""
    
    def __init__(self):
        # httpbin.org 路径映射
        self.path_map = {
            'GET': '/get',
            'POST': '/post', 
            'PUT': '/put',
            'DELETE': '/delete',
            'PATCH': '/patch'
        }
    
    def generate(self, api_desc: str, case_type: str = 'normal') -> dict:
        """
        根据API描述和用例类型生成测试用例
        
        Args:
            api_desc: API描述文本
            case_type: 'normal'(正常), 'boundary'(边界), 'exception'(异常)
        """
        # 解析HTTP方法
        method = self._parse_method(api_desc)
        path = self.path_map.get(method, '/anything')
        
        # 根据case_type生成不同的测试值
        test_data = self._generate_test_data(method, case_type)
        
        # 构建请求
        request = {
            "method": method,
            "url": path,
            "headers": {},
            "query": {},
            "body": None
        }
        
        # 根据方法放置参数
        if method in ['GET', 'DELETE']:
            request['query'] = test_data
        else:
            request['body'] = test_data
            request['headers']['Content-Type'] = 'application/json'
        
        return {
            "description": f"{method} {path} - {case_type} 用例",
            "request": request,
            "expected_response": {
                "status_code": 200,
                "description": self._get_expected_desc(case_type)
            },
            "_case_type": case_type,
            "_source": "rule"
        }
    
    def _parse_method(self, api_desc: str) -> str:
        """解析HTTP方法"""
        match = re.search(r'(GET|POST|PUT|DELETE|PATCH)', api_desc, re.IGNORECASE)
        return match.group(1).upper() if match else 'GET'
    
    def _generate_test_data(self, method: str, case_type: str) -> dict:
        """
        根据用例类型生成测试数据
        """
        if case_type == 'normal':
            # 正常值：合法、典型的数据
            return {
                "id": "1",
                "name": "test_user",
                "email": "user@example.com",
                "age": 25,
                "status": "active"
            }
        
        elif case_type == 'boundary':
            # 边界值：临界条件、空值、极值
            return {
                "id": "",  # 空字符串（边界）
                "name": "a",  # 最小长度
                "email": "a@b.c",  # 最短邮箱
                "age": 0,  # 最小值
                "status": ""  # 空值
            }
        
        else:  # exception
            # 异常值：非法格式、注入攻击、类型错误
            return {
                "id": "'; DROP TABLE users; --",  # SQL注入
                "name": "<script>alert('xss')</script>",  # XSS攻击
                "email": "not-an-email",  # 非法格式
                "age": "not_a_number",  # 类型错误
                "status": "A" * 10000  # 超长字符串
            }
    
    def _get_expected_desc(self, case_type: str) -> str:
        """获取预期响应描述"""
        descriptions = {
            'normal': '正常请求，应返回200',
            'boundary': '边界值测试，应返回200或400',
            'exception': '异常值测试，应返回400错误'
        }
        return descriptions.get(case_type, '未知')


class ModelGenerator:
    """基于微调模型的生成器"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE)
            self.model.eval()
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"⚠️  模型加载失败: {e}")
    
    def generate(self, api_desc: str, case_type: str = 'normal') -> dict:
        """使用模型生成"""
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        # 构建带类型标记的prompt
        prompt = f"[{case_type.upper()}] {api_desc}\n生成JSON格式测试用例:"
        
        # 生成
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                do_sample=False
            )
        
        raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 解析
        return self._parse_output(raw, case_type)
    
    def _parse_output(self, raw: str, case_type: str) -> dict:
        """解析模型输出"""
        raw = raw.strip()
        
        # 尝试直接解析
        try:
            result = json.loads(raw)
            result['_case_type'] = case_type
            result['_source'] = 'model'
            return result
        except:
            pass
        
        # 尝试提取JSON
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1:
            try:
                result = json.loads(raw[start:end+1])
                result['_case_type'] = case_type
                result['_source'] = 'model_extracted'
                return result
            except:
                pass
        
        # 解析失败
        raise ValueError(f"无法解析模型输出: {raw[:100]}")


class HybridGenerator:
    """混合生成器：模型为主，规则兜底"""
    
    def __init__(self):
        self.model_gen = ModelGenerator()
        self.rule_gen = RuleBasedGenerator()
        self.use_model = self.model_gen.model is not None
    
    def generate(self, api_desc: str, case_type: str = 'normal') -> dict:
        """
        生成单个测试用例
        
        优先使用模型，失败则回退到规则
        """
        # 尝试模型生成
        if self.use_model:
            try:
                result = self.model_gen.generate(api_desc, case_type)
                # 验证结果完整性
                if self._validate_result(result):
                    return result
            except Exception as e:
                print(f"  ⚠️  模型生成失败 [{case_type}]: {e}")
        
        # 回退到规则
        result = self.rule_gen.generate(api_desc, case_type)
        result['_source'] = 'rule_fallback'
        return result
    
    def generate_all_types(self, api_desc: str) -> dict:
        """
        为单个API生成三类测试用例
        
        Returns:
            {'normal': {...}, 'boundary': {...}, 'exception': {...}}
        """
        results = {}
        case_types = ['normal', 'boundary', 'exception']
        
        print(f"\n处理API: {api_desc[:50]}...")
        
        for case_type in case_types:
            print(f"  生成 [{case_type}]...", end=" ")
            result = self.generate(api_desc, case_type)
            
            source = result.get('_source', 'unknown')
            if 'model' in source:
                print(f"✅ 模型 ({source})")
            else:
                print(f"⚠️  规则 ({source})")
            
            results[case_type] = result
        
        return results
    
    def _validate_result(self, result: dict) -> bool:
        """验证生成结果是否有效"""
        required_keys = ['description', 'request']
        return all(key in result for key in required_keys)


# ========== 便捷函数 ==========

def create_generator():
    """创建生成器实例"""
    return HybridGenerator()


def generate_test_case(api_desc: str, case_type: str = 'normal', generator=None):
    """
    便捷函数：生成单个测试用例
    
    Args:
        api_desc: API描述
        case_type: 用例类型
        generator: 可选，传入已有生成器实例
    
    Returns:
        测试用例字典
    """
    if generator is None:
        generator = create_generator()
    return generator.generate(api_desc, case_type)


# ========== 主函数 ==========

def main():
    """测试"""
    print("=" * 60)
    print("混合生成器 v