"""
API文档解析模块
支持解析OpenAPI 3.0 JSON/YAML格式的文档，提取端点信息和参数约束
"""
import json
import yaml
import os
from typing import Dict, List, Any, Optional

class OpenAPIParser:
    """OpenAPI文档解析器"""
    
    def __init__(self, spec_path: str = None):
        """
        初始化解析器，可选择直接加载文档
        Args:
            spec_path: OpenAPI文档文件路径（JSON或YAML）
        """
        self.spec = None
        if spec_path:
            self.load(spec_path)
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        加载并解析OpenAPI文档
        Args:
            file_path: 文档路径
        Returns:
            解析后的API规范字典
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if ext in ['.json']:
            self.spec = json.loads(content)
        elif ext in ['.yaml', '.yml']:
            self.spec = yaml.safe_load(content)
        else:
            raise ValueError(f"不支持的文件格式: {ext}，仅支持 .json .yaml .yml")
        
        return self.spec
        self.spec = self.spec   # 将加载的规范保存为实例属性
    
    def extract_endpoints(self) -> List[Dict[str, Any]]:
        """
        从已加载的规范中提取所有端点信息
        Returns:
            端点信息列表，每个元素包含 method, path, summary, parameters, requestBody 等
        """
        if not self.spec:
            raise ValueError("请先调用 load() 加载文档")
        
        endpoints = []
        paths = self.spec.get('paths', {})
        
        for path, methods in paths.items():
            for method, details in methods.items():
                method_upper = method.upper()
                if method_upper not in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']:
                    continue  # 跳过非标准方法
                
                # 提取参数信息
                parameters = details.get('parameters', [])
                
                # 提取请求体
                request_body = details.get('requestBody', {})
                
                # 生成人类可读的描述
                summary = details.get('summary', '')
                description = details.get('description', '')
                
                endpoint_info = {
                    'path': path,
                    'method': method_upper,
                    'summary': summary,
                    'description': description,
                    'parameters': self._normalize_parameters(parameters),
                    'requestBody': self._normalize_request_body(request_body)
                }
                endpoints.append(endpoint_info)
        
        return endpoints
    
    def _normalize_parameters(self, params: List[Dict]) -> List[Dict]:
        """规范化参数格式，提取约束信息"""
        normalized = []
        for p in params:
            schema = p.get('schema', {})
            normalized.append({
                'name': p.get('name'),
                'in': p.get('in'),  # query, path, header, cookie
                'required': p.get('required', False),
                'type': schema.get('type'),
                'format': schema.get('format'),
                'description': p.get('description', ''),
                'constraints': {
                    'minimum': schema.get('minimum'),
                    'maximum': schema.get('maximum'),
                    'minLength': schema.get('minLength'),
                    'maxLength': schema.get('maxLength'),
                    'pattern': schema.get('pattern'),
                    'enum': schema.get('enum')
                }
            })
        return normalized
    
    def _normalize_request_body(self, rb: Dict) -> Optional[Dict]:
        """规范化请求体信息"""
        if not rb:
            return None
        content = rb.get('content', {})
        # 通常取 application/json 的 schema
        json_schema = None
        if 'application/json' in content:
            json_schema = content['application/json'].get('schema')
        elif 'application/x-www-form-urlencoded' in content:
            json_schema = content['application/x-www-form-urlencoded'].get('schema')
        
        return {
            'required': rb.get('required', False),
            'description': rb.get('description', ''),
            'schema': json_schema
        }
    
    def generate_description(self, endpoint: Dict) -> str:
        """
        为端点生成自然语言描述，供模型输入使用
        Args:
            endpoint: extract_endpoints 返回的单个端点信息
        Returns:
            描述字符串
        """
        parts = [f"{endpoint['method']} {endpoint['path']}"]
        if endpoint['summary']:
            parts.append(f"- {endpoint['summary']}")
        
        # 参数描述
        if endpoint['parameters']:
            param_desc = []
            for p in endpoint['parameters']:
                desc = f"{p['name']} ({p['type']}"
                if p['required']:
                    desc += ", 必填"
                else:
                    desc += ", 可选"
                # 添加约束
                cons = p['constraints']
                if cons['minimum'] is not None:
                    desc += f", 最小值={cons['minimum']}"
                if cons['maximum'] is not None:
                    desc += f", 最大值={cons['maximum']}"
                if cons['minLength'] is not None:
                    desc += f", 最小长度={cons['minLength']}"
                if cons['maxLength'] is not None:
                    desc += f", 最大长度={cons['maxLength']}"
                if cons['enum']:
                    desc += f", 枚举值={cons['enum']}"
                desc += ")"
                param_desc.append(desc)
            parts.append("参数: " + ", ".join(param_desc))
        
        # 请求体描述
        if endpoint['requestBody'] and endpoint['requestBody']['schema']:
            schema = endpoint['requestBody']['schema']
            if schema.get('type') == 'object':
                props = schema.get('properties', {})
                required_list = schema.get('required', [])
                prop_desc = []
                for name, prop in props.items():
                    req = "必填" if name in required_list else "可选"
                    prop_type = prop.get('type', 'unknown')
                    prop_desc.append(f"{name} ({prop_type}, {req})")
                if prop_desc:
                    parts.append("请求体: " + ", ".join(prop_desc))
        
        return " ".join(parts)


# 测试代码
if __name__ == "__main__":
    # 使用临时文件测试YAML解析
    import tempfile
    sample_yaml = """
openapi: 3.0.0
info:
  title: 用户管理API
  version: 1.0.0
paths:
  /users:
    get:
      summary: 获取用户列表
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            minimum: 1
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            minimum: 1
            maximum: 100
      responses:
        '200':
          description: 成功
    post:
      summary: 创建新用户
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [username]
              properties:
                username:
                  type: string
                  minLength: 3
                  maxLength: 20
                age:
                  type: integer
                  minimum: 0
                  maximum: 120
      responses:
        '201':
          description: 创建成功
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        f.write(sample_yaml)
        temp_path = f.name
    
    parser = OpenAPIParser(temp_path)
    endpoints = parser.extract_endpoints()
    print(f"解析到 {len(endpoints)} 个端点：")
    for ep in endpoints:
        desc = parser.generate_description(ep)
        print(f"  {desc}")
    
    # 清理临时文件
    os.unlink(temp_path)
    print("\n✅ 解析器测试通过！")