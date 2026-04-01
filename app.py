#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API测试用例生成系统 - 现代化界面
优化版：专业工具风格，增加可配置的测试参数
"""

import streamlit as st
import json
import pandas as pd
import time
import sys
import os
import tempfile
import re
from datetime import datetime
from typing import List, Dict, Optional
import plotly.express as px
import plotly.graph_objects as go

sys.path.append('src')

# 尝试导入自定义模块
try:
    from hybrid_generator import load_model, generate_test_case
    from test_runner import TestRunner
    CUSTOM_MODULES_AVAILABLE = True
except ImportError:
    CUSTOM_MODULES_AVAILABLE = False
    st.warning("⚠️ 自定义模块未找到，将使用模拟数据演示")

# ========== 页面配置 ==========
st.set_page_config(
    page_title="API TestCraft - 智能测试用例生成平台",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== 自定义CSS样式 ==========
def load_custom_css():
    st.markdown("""
    <style>
        /* 全局字体和颜色 */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* 主色调：科技蓝 + 深灰 */
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --secondary: #64748b;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
        }
        
        /* 顶部导航栏 */
        .main-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            padding: 1rem 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            border: 1px solid #334155;
        }
        
        .header-title {
            color: #f8fafc;
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .header-subtitle {
            color: #94a3b8;
            font-size: 0.875rem;
            margin-top: 0.25rem;
        }
        
        /* 标签页样式 */
        .stTabs [data-baseweb="tab-list"] {
            background: #1e293b;
            border-radius: 10px;
            padding: 0.5rem;
            gap: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #94a3b8;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .stTabs [aria-selected="true"] {
            background: #2563eb !important;
            color: white !important;
        }
        
        /* 卡片样式 */
        .metric-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.5rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 40px -10px rgba(37, 99, 235, 0.2);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #f8fafc;
        }
        
        .metric-label {
            color: #94a3b8;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* 状态指示器 */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.375rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .status-success {
            background: rgba(16, 185, 129, 0.1);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }
        
        .status-warning {
            background: rgba(245, 158, 11, 0.1);
            color: #f59e0b;
            border: 1px solid rgba(245, 158, 11, 0.2);
        }
        
        .status-error {
            background: rgba(239, 68, 68, 0.1);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.2);
        }
        
        /* 代码块样式 */
        .code-block {
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1rem;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.875rem;
            overflow-x: auto;
        }
        
        .code-keyword { color: #c084fc; }
        .code-string { color: #4ade80; }
        .code-number { color: #fbbf24; }
        
        /* 按钮样式 */
        .stButton > button {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.2s;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }
        
        /* 输入框样式 */
        .stTextArea > div > div {
            background: #f3f4f6;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            color: #1e293b;
        }
        
        .stTextArea > div > div:focus-within {
            border-color: #2563eb;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }
        
        /* 滑块样式 */
        .stSlider > div > div {
            color: #2563eb;
        }
        
        /* 表格样式 */
        .dataframe {
            background: #1e293b;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .dataframe th {
            background: #0f172a;
            color: #f8fafc;
            font-weight: 600;
            padding: 1rem;
        }
        
        .dataframe td {
            padding: 0.875rem 1rem;
            border-bottom: 1px solid #334155;
        }
        
        /* 进度条 */
        .progress-container {
            background: #1e293b;
            border-radius: 9999px;
            height: 8px;
            overflow: hidden;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
            border-radius: 9999px;
            transition: width 0.3s ease;
        }
        
        /* 隐藏默认Streamlit元素 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# ========== 状态管理 ==========
if 'generated_cases' not in st.session_state:
    st.session_state.generated_cases = []
if 'execution_results' not in st.session_state:
    st.session_state.execution_results = []
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "生成"
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# ========== 顶部导航栏 ==========
def render_header():
    col1, col2, col3 = st.columns([6, 2, 2])
    
    with col1:
        st.markdown("""
        <div class="main-header">
            <h1 class="header-title">
                <span>🧪</span>
                API TestCraft
            </h1>
            <p class="header-subtitle">
                基于深度学习的智能测试用例生成与执行平台 | 
                <span class="status-badge status-success">
                    <span>●</span> 系统就绪
                </span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.model_loaded:
            st.success("🤖 模型已加载")
        else:
            st.info("⏳ 等待加载")
    
    with col3:
        if st.button("🔄 重置会话", key="reset_btn"):
            st.session_state.clear()
            st.rerun()

# ========== API输入组件 ==========
def render_api_input():
    st.markdown("### 📝 API描述输入")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("#### 或上传 OpenAPI 文件")
        uploaded_file = st.file_uploader("选择 OpenAPI 文件 (JSON/YAML)", type=['json', 'yaml', 'yml'], key="openapi_file")
        templates = {
            "自定义输入": "",
            "用户创建": "生成测试用例JSON: POST /users - 创建用户，需要username（字符串，必填，3-20字符）和age（整数，可选，0-120）",
            "用户信息": "生成测试用例JSON: GET /users/{id} - 获取用户信息，id为整数",
            "用户删除": "生成测试用例JSON: DELETE /users/{id} - 删除用户，id为整数",
        }
        selected_template = st.selectbox(
            "📋 选择预设模板",
            options=list(templates.keys()),
            key="template_select"
        )
        api_description = st.text_area(
            "API描述（支持自然语言或结构化格式）",
            value=templates[selected_template] if selected_template != "自定义输入" else "",
            height=120,
            placeholder="例如：生成测试用例JSON: POST /login 参数：username(string,必填), password(string,必填)",
            key="api_desc"
        )
    
    with col2:
        st.markdown("#### ⚙️ 生成选项")
        case_types = st.multiselect(
            "用例类型",
            options=[
                ("normal", "✅ 正常值"),
                ("boundary", "⚠️ 边界值"),
                ("exception", "❌ 异常值")
            ],
            format_func=lambda x: x[1],
            default=[("normal", "✅ 正常值"), ("boundary", "⚠️ 边界值"), ("exception", "❌ 异常值")],
            key="case_types"
        )
        num_variants = st.slider(
            "每种类型生成数量",
            min_value=1,
            max_value=5,
            value=2,
            key="num_variants"
        )
        use_hybrid = st.toggle(
            "启用混合生成（模型+规则）",
            value=True,
            key="use_hybrid",
            help="当模型输出不稳定时自动切换到规则引擎"
        )
        generate_btn = st.button(
            "🚀 生成测试用例",
            type="primary",
            use_container_width=True,
            key="generate_btn"
        )
    return api_description, [ct[0] for ct in case_types], num_variants, use_hybrid, generate_btn, uploaded_file

# ========== 测试用例生成 ==========
def generate_mock_case(api_desc: str, case_type: str, variant: int) -> Dict:
    """生成模拟测试用例"""
    methods = ["GET", "POST", "PUT", "DELETE"]
    method = methods[hash(api_desc) % len(methods)]
    base_case = {
        "description": f"{case_type.upper()} 测试用例 #{variant+1}",
        "case_type": case_type,
        "request": {
            "method": method,
            "url": f"https://httpbin.org/{method.lower()}",
            "headers": {"Content-Type": "application/json"},
            "body": None
        },
        "expected_response": {
            "status_code": 200 if case_type == "normal" else (400 if case_type == "exception" else 200),
            "description": "预期响应"
        }
    }
    if case_type == "normal":
        base_case["request"]["body"] = {"username": f"user_{variant}", "age": 25 + variant}
    elif case_type == "boundary":
        base_case["request"]["body"] = {"username": "a" * 20, "age": 120 if variant % 2 else 0}
    else:
        base_case["request"]["body"] = {"username": None, "age": -999}
    return base_case

def resolve_ref(schema, components):
    """递归解析 $ref，返回实际 schema 对象"""
    if '$ref' in schema:
        ref = schema['$ref']
        parts = ref.split('/')
        if parts[1] == 'components' and parts[2] == 'schemas':
            schema_name = parts[3]
            return components.get('schemas', {}).get(schema_name, {})
    return schema

def generate_cases(api_desc: str, case_types: List[str], num_variants: int, use_hybrid: bool, uploaded_file):
    """生成测试用例（支持文件上传，基于 OpenAPI schema 动态生成，支持嵌套对象）"""
    cases = []
    
    # 加载模型（如果未加载且可用）
    if not st.session_state.model_loaded and CUSTOM_MODULES_AVAILABLE:
        with st.spinner("🤖 正在加载AI模型..."):
            try:
                tokenizer, model = load_model()
                st.session_state.tokenizer = tokenizer
                st.session_state.model = model
                st.session_state.model_loaded = True
            except Exception as e:
                st.error(f"模型加载失败: {e}")
                st.session_state.model_loaded = False
    
    # ========== 递归生成 schema 值的辅助函数 ==========
    def generate_value_from_schema(schema, case_type, components, field_name=None):
        """
        根据 JSON Schema 和用例类型生成测试值，支持递归处理嵌套对象。
        schema: 字段的 schema 对象
        case_type: 'normal'/'boundary'/'exception'
        components: 整个 OpenAPI 的 components 对象（用于解析引用）
        """
        # 先解析可能的 $ref
        schema = resolve_ref(schema, components)
        stype = schema.get('type')
        
        # 处理枚举
        if 'enum' in schema:
            enums = schema['enum']
            if case_type == 'normal':
                return enums[0] if enums else None
            elif case_type == 'boundary':
                return enums[-1] if enums else None
            else:
                if stype == 'string':
                    return 'invalid_enum'
                else:
                    return 9999

        # 处理对象类型（递归生成）
        if stype == 'object' and 'properties' in schema:
            obj = {}
            for prop_name, prop_schema in schema['properties'].items():
                obj[prop_name] = generate_value_from_schema(prop_schema, case_type, components, prop_name)
            return obj

        # 处理数组类型
        if stype == 'array':
            items = schema.get('items', {})
            return [generate_value_from_schema(items, case_type, components, field_name)]

        # 处理基本类型
        if stype == 'string':
            max_len = schema.get('maxLength')
            min_len = schema.get('minLength', 1)
            if case_type == 'normal':
                if max_len and min_len:
                    length = (min_len + max_len) // 2
                else:
                    length = 8
                return 't' * length
            elif case_type == 'boundary':
                if max_len:
                    return 'a' * max_len
                else:
                    return 'a' * 100
            else:
                return 12345  # 类型错误

        if stype == 'integer':
            minimum = schema.get('minimum')
            maximum = schema.get('maximum')
            if case_type == 'normal':
                if minimum is not None and maximum is not None:
                    return (minimum + maximum) // 2
                elif minimum is not None:
                    return minimum + 5
                elif maximum is not None:
                    return maximum - 5
                else:
                    return 25
            elif case_type == 'boundary':
                if minimum is not None and maximum is not None:
                    return minimum if (hash(str(field_name)) % 2 == 0) else maximum
                elif minimum is not None:
                    return minimum
                elif maximum is not None:
                    return maximum
                else:
                    return 999999
            else:
                return 'not_a_number'

        if stype == 'boolean':
            if case_type == 'normal':
                return True
            elif case_type == 'boundary':
                return False
            else:
                return 'not_boolean'

        if stype == 'number':
            if case_type == 'normal':
                return 12.34
            elif case_type == 'boundary':
                return 9999.99
            else:
                return 'not_a_number'

        return 'test_value'

    # ========== 文件上传模式 ==========
    if uploaded_file is not None:
        import tempfile
        from api_parser import OpenAPIParser
        
        # 保存临时文件
        suffix = '.' + uploaded_file.name.split('.')[-1]
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False, encoding='utf-8') as f:
            f.write(uploaded_file.getvalue().decode('utf-8'))
            temp_path = f.name
        
        # 解析文档
        parser = OpenAPIParser(temp_path)
        try:
            spec = parser.spec
        except AttributeError:
            import json, yaml
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if suffix == '.json':
                spec = json.loads(content)
            else:
                spec = yaml.safe_load(content)
        components = spec.get('components', {})
        
        endpoints = parser.extract_endpoints()
        os.unlink(temp_path)
        
        for ep in endpoints:
            method = ep['method']
            path = ep['path']
            description = ep.get('description', '')
            path_params = [p for p in ep.get('parameters', []) if p.get('in') == 'path']
            request_body_schema = None
            if ep.get('requestBody') and ep['requestBody'].get('schema'):
                schema = ep['requestBody']['schema']
                schema = resolve_ref(schema, components)
                if schema.get('type') == 'object' and 'properties' in schema:
                    request_body_schema = schema
            
            for case_type in case_types:
                for i in range(num_variants):
                    body = {}
                    if request_body_schema:
                        body = generate_value_from_schema(request_body_schema, case_type, components)
                    
                    url = path
                    for p in path_params:
                        p_name = p['name']
                        if case_type == 'normal':
                            val = 1
                        elif case_type == 'boundary':
                            val = 999999
                        else:
                            val = -1
                        url = url.replace(f'{{{p_name}}}', str(val))
                    
                    if case_type == 'normal':
                        if method == 'POST':
                            expected_status = 201
                        elif method == 'DELETE':
                            expected_status = 204
                        else:
                            expected_status = 200
                    else:
                        if method == 'POST':
                            expected_status = 400
                        elif method in ['GET', 'DELETE'] and path_params:
                            expected_status = 404
                        else:
                            expected_status = 400
                    
                    test_case = {
                        "description": f"{method} {path} - {case_type} test",
                        "case_type": case_type,
                        "request": {
                            "method": method,
                            "url": url,
                            "headers": {
                                "Content-Type": "application/json",
                                "Accept": "application/json"
                            },
                            "query": {},
                            "body": body if method in ["POST", "PUT", "PATCH"] else {}
                        },
                        "expected_response": {
                            "status_code": expected_status,
                            "headers": {"Content-Type": "application/json"}
                        }
                    }
                    test_case['_case_type'] = case_type
                    test_case['_variant'] = i + 1
                    test_case['_id'] = f"{case_type}_{i+1:02d}"
                    test_case['_display_id'] = f"{case_type}_{i+1:02d}"
                    cases.append(test_case)
        return cases
    
    # ========== 手动输入模式 ==========
    if not api_desc:
        return cases
    
    # 解析描述，提取方法、路径和描述文本
    def parse_manual_desc(desc):
        desc = re.sub(r'^\[[A-Z]+\]\s*', '', desc)
        patterns = [
            r'(?:生成测试用例JSON:)?\s*(\w+)\s+(/[^\s-]+)\s*-\s*(.+)',
            r'(?:生成测试用例JSON:)?\s*(\w+)\s+(/[^\s]+)\s+(.+)',
            r'(?:生成测试用例JSON:)?\s*(\w+)\s+(/[^\s]+)',
        ]
        method, path, desc_text = "GET", "/unknown", "test"
        for pattern in patterns:
            match = re.match(pattern, desc, re.IGNORECASE)
            if match:
                method = match.group(1).upper()
                path = match.group(2)
                desc_text = match.group(3) if len(match.groups()) > 2 else ""
                break
        return method, path, desc_text
    
    def extract_path_params(path):
        return re.findall(r'\{(\w+)\}', path)
    
    method, path, desc_text = parse_manual_desc(api_desc)
    path_params = extract_path_params(path)
    
    for case_type in case_types:
        for i in range(num_variants):
            # 构造带类型前缀的描述，传给模型
            typed_desc = f"[{case_type.upper()}] {api_desc}"
            
            # 调用模型生成用例（如果模型可用）
            if CUSTOM_MODULES_AVAILABLE and st.session_state.model_loaded:
                try:
                    case = generate_test_case(typed_desc, st.session_state.tokenizer, st.session_state.model)
                except Exception as e:
                    st.warning(f"模型生成失败: {e}，使用规则回退")
                    case = generate_mock_case(api_desc, case_type, i)
            else:
                case = generate_mock_case(api_desc, case_type, i)
            
            # 生成路径参数值（用于替换 URL）
            url = path
            for param in path_params:
                if case_type == 'normal':
                    val = 1
                elif case_type == 'boundary':
                    val = 999999
                else:  # exception
                    val = -1
                url = url.replace(f'{{{param}}}', str(val))
            
            # 覆盖 URL
            case['request']['url'] = url
            
            # 调整期望状态码
            if method in ['GET', 'DELETE'] and path_params:
                if case_type == 'normal':
                    expected_status = 200 if method == 'GET' else 204
                else:
                    expected_status = 404
            else:
                # 对于无路径参数的请求，保持模型/规则生成的期望状态码
                expected_status = case.get('expected_response', {}).get('status_code', 200)
            case['expected_response']['status_code'] = expected_status
            
            # 在 cases.append(case) 之前添加
            case['expected_response']['status_code'] = expected_status

            # 统一描述字段
            case['description'] = f"{method} {path} - {case_type} test"

            # 统一类型字段
            actual_type = case_type
            case['case_type'] = actual_type
            case['_case_type'] = actual_type
            case['_variant'] = i + 1
            case['_id'] = f"{actual_type}_{i+1:02d}"
            case['_display_id'] = f"{actual_type}_{i+1:02d}"

            cases.append(case)
    
    return cases

# ========== 用例展示组件 ==========
def render_case_cards(cases: List[Dict]):
    if not cases:
        st.info("💡 点击左侧'生成测试用例'按钮开始")
        return
    
    st.markdown("### 📊 生成概览")
    cols = st.columns(4)
    
    def get_case_type(case):
        return case.get('case_type', case.get('_case_type', 'unknown'))
    
    total = len(cases)
    normal_count = sum(1 for c in cases if get_case_type(c) == 'normal')
    boundary_count = sum(1 for c in cases if get_case_type(c) == 'boundary')
    exception_count = sum(1 for c in cases if get_case_type(c) == 'exception')
    
    metrics = [
        ("总用例数", total, "📋"),
        ("正常值", normal_count, "✅"),
        ("边界值", boundary_count, "⚠️"),
        ("异常值", exception_count, "❌")
    ]
    for col, (label, value, icon) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{icon} {label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 用例详情")
    filter_col1, filter_col2 = st.columns([1, 3])
    with filter_col1:
        type_filter = st.selectbox("筛选类型", options=["全部", "normal", "boundary", "exception"], key="type_filter")
    
    filtered_cases = cases if type_filter == "全部" else [c for c in cases if get_case_type(c) == type_filter]
    
    df_data = []
    for idx, c in enumerate(filtered_cases):
        req = c.get('request', {})
        display_id = c.get('_display_id', f"case_{idx+1:02d}")
        case_type_display = {
            'normal': '✅ 正常',
            'boundary': '⚠️ 边界',
            'exception': '❌ 异常'
        }.get(get_case_type(c), get_case_type(c))
        
        df_data.append({
            "ID": display_id,
            "类型": case_type_display,
            "方法": req.get('method', '-'),
            "URL": req.get('url', '-')[:50],
            "预期状态": c.get('expected_response', {}).get('status_code', '-'),
        })
    df = pd.DataFrame(df_data)
    st.dataframe(df, width='stretch', height=400)
    
    options = [c.get('_display_id', f"case_{idx+1:02d}") for idx, c in enumerate(filtered_cases)]
    selected_display_id = st.selectbox(
        "查看详细用例",
        options=options,
        key="case_detail_select"
    )
    if selected_display_id:
        case = next(c for c in filtered_cases if c.get('_display_id') == selected_display_id)
        with st.expander("📄 完整JSON", expanded=True):
            st.json(case)

# ========== 测试执行（带配置）==========
def execute_tests(cases: List[Dict], base_url: str, timeout: int, retries: int, delay: float) -> List[Dict]:
    """执行测试用例（使用可配置的执行器）"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if CUSTOM_MODULES_AVAILABLE:
        try:
            runner = TestRunner(base_url=base_url, timeout=timeout, retries=retries, delay=delay)
        except Exception as e:
            st.error(f"测试执行器初始化失败: {e}")
            runner = None
    else:
        runner = None
    
    for i, case in enumerate(cases):
        progress = (i + 1) / len(cases)
        progress_bar.progress(progress)
        status_text.text(f"执行中 {i+1}/{len(cases)}: {case.get('_id', 'unknown')}...")
        
        if runner:
            try:
                result = runner.run([case])[0]
            except Exception as e:
                st.warning(f"用例 {case.get('_id', '')} 执行失败: {e}，使用模拟结果")
                result = mock_execute(case)
        else:
            result = mock_execute(case)
        
        results.append(result)
        time.sleep(0.2)
    
    progress_bar.empty()
    status_text.empty()
    return results

def mock_execute(case: Dict) -> Dict:
    """模拟测试执行"""
    import random
    case_type = case.get('_case_type', 'normal')
    if case_type == 'normal':
        passed = random.random() > 0.1
        actual_status = 200 if passed else 500
    elif case_type == 'boundary':
        passed = random.random() > 0.3
        actual_status = 200 if passed else 400
    else:
        passed = random.random() > 0.5
        actual_status = 400 if passed else 200
    return {
        'test_case': case,
        'request': case.get('request', {}),
        'response': {
            'status_code': actual_status,
            'elapsed': random.uniform(0.1, 2.0),
            'body': {'message': 'mock response'}
        },
        'expected_status': case.get('expected_response', {}).get('status_code', 200),
        'passed': passed,
        'timestamp': datetime.now().isoformat()
    }

# ========== 结果可视化 ==========
def render_results(results: List[Dict]):
    if not results:
        st.info("请先执行测试")
        return
    
    st.markdown("### 📈 执行统计")
    total = len(results)
    passed = len([r for r in results if r['passed']])
    failed = total - passed
    avg_time = sum(r['response']['elapsed'] for r in results) / total
    
    cols = st.columns(4)
    metrics = [
        ("总执行数", total, "📋", "#3b82f6"),
        ("通过", passed, "✅", "#10b981"),
        ("失败", failed, "❌", "#ef4444"),
        ("平均耗时", f"{avg_time:.2f}s", "⏱️", "#f59e0b")
    ]
    for col, (label, value, icon, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {color}">
                <div class="metric-label">{icon} {label}</div>
                <div class="metric-value" style="color: {color}">{value}</div>
            </div>
            """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### 通过率分布")
        fig_pie = go.Figure(data=[go.Pie(
            labels=['通过', '失败'],
            values=[passed, failed],
            hole=0.4,
            marker_colors=['#10b981', '#ef4444']
        )])
        fig_pie.update_layout(
            showlegend=True,
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("#### 响应时间分布")
        type_times = {}
        for r in results:
            t = r['test_case'].get('_case_type', 'unknown')
            type_times.setdefault(t, []).append(r['response']['elapsed'])
        fig_bar = go.Figure()
        for t, times in type_times.items():
            fig_bar.add_trace(go.Bar(
                name=t,
                x=[t],
                y=[sum(times)/len(times)],
                marker_color={
                    'normal': '#3b82f6',
                    'boundary': '#f59e0b',
                    'exception': '#ef4444'
                }.get(t, '#64748b')
            ))
        fig_bar.update_layout(
            barmode='group',
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc',
            xaxis_title="用例类型",
            yaxis_title="平均响应时间 (s)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("### 📋 详细结果")
    df_data = []
    for r in results:
        tc = r['test_case']
        status_badge = "✅ 通过" if r['passed'] else "❌ 失败"
        df_data.append({
            "用例ID": tc.get('_display_id', tc.get('_id', 'unknown')),
            "类型": tc.get('case_type', tc.get('_case_type', 'unknown')),
            "描述": tc.get('description', '-')[:40],
            "方法": r['request'].get('method', '-'),
            "期望状态": r['expected_status'],
            "实际状态": r['response']['status_code'],
            "耗时(s)": round(r['response']['elapsed'], 3),
            "结果": status_badge
        })
    df = pd.DataFrame(df_data)
    st.dataframe(df, width='stretch', height=400)
    
    st.markdown("### 💾 导出报告")
    report_data = {
        'generated_at': datetime.now().isoformat(),
        'summary': {
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': f"{passed/total*100:.1f}%"
        },
        'results': results
    }
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "📥 下载JSON报告",
            data=json.dumps(report_data, ensure_ascii=False, indent=2),
            file_name=f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    with col_dl2:
        html_report = generate_html_report(report_data)
        st.download_button(
            "📄 下载HTML报告",
            data=html_report,
            file_name=f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True
        )

def generate_html_report(data: Dict) -> str:
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>API测试报告 - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; background: #f8fafc; }}
            .header {{ background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); color: white; padding: 2rem; border-radius: 12px; margin-bottom: 2rem; }}
            .metric {{ display: inline-block; background: white; padding: 1rem 2rem; border-radius: 8px; margin: 0.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .metric-value {{ font-size: 2rem; font-weight: bold; color: #1e293b; }}
            .metric-label {{ color: #64748b; font-size: 0.875rem; }}
            table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; }}
            th {{ background: #1e293b; color: white; padding: 1rem; text-align: left; }}
            td {{ padding: 0.875rem 1rem; border-bottom: 1px solid #e2e8f0; }}
            .pass {{ color: #10b981; }}
            .fail {{ color: #ef4444; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧪 API测试执行报告</h1>
            <p>生成时间: {data['generated_at']}</p>
        </div>
        <div style="text-align: center; margin-bottom: 2rem;">
            <div class="metric">
                <div class="metric-value">{data['summary']['total']}</div>
                <div class="metric-label">总用例</div>
            </div>
            <div class="metric">
                <div class="metric-value" style="color: #10b981;">{data['summary']['passed']}</div>
                <div class="metric-label">通过</div>
            </div>
            <div class="metric">
                <div class="metric-value" style="color: #ef4444;">{data['summary']['failed']}</div>
                <div class="metric-label">失败</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data['summary']['pass_rate']}</div>
                <div class="metric-label">通过率</div>
            </div>
        </div>
        <h2>详细结果</h2>
         膝
            <thead>
                <tr>
                    <th>用例ID</th>
                    <th>类型</th>
                    <th>描述</th>
                    <th>方法</th>
                    <th>期望状态</th>
                    <th>实际状态</th>
                    <th>耗时(s)</th>
                    <th>结果</th>
                 表示
            </thead>
            <tbody>
    """
    for r in data['results']:
        tc = r['test_case']
        status_class = 'pass' if r['passed'] else 'fail'
        status_text = '✅ 通过' if r['passed'] else '❌ 失败'
        html += f"""
                 <tr>
                      <td>{tc.get('_id', '-')}</td>
                      <td>{tc.get('_case_type', '-')}</td>
                      <td>{tc.get('description', '-')[:50]}</td>
                      <td>{r['request'].get('method', '-')}</td>
                      <td>{r['expected_status']}</td>
                      <td>{r['response']['status_code']}</td>
                      <td>{r['response']['elapsed']:.3f}</td>
                      <td class="{status_class}">{status_text}</td>
                  </tr>
        """
    html += """
            </tbody>
         </table>
    </body>
    </html>
    """
    return html

# ========== 历史记录 ==========
def render_history():
    st.markdown("### 📚 历史记录")
    if 'history' not in st.session_state:
        st.session_state.history = []
    history = st.session_state.history
    if not history:
        st.info("暂无历史记录，开始生成你的第一个测试用例吧！")
        return
    for i, record in enumerate(reversed(history[-10:])):
        with st.container():
            col1, col2, col3 = st.columns([2, 4, 2])
            with col1:
                st.markdown(f"**{record['timestamp'][:16]}**")
                st.caption(f"{record['api_desc'][:30]}...")
            with col2:
                metrics = record.get('metrics', {})
                st.markdown(f"📋 {metrics.get('total', 0)} 用例 | ✅ {metrics.get('passed', 0)} 通过 | ❌ {metrics.get('failed', 0)} 失败")
            with col3:
                if st.button("📂 加载", key=f"load_hist_{i}"):
                    st.session_state.generated_cases = record['cases']
                    st.session_state.execution_results = record['results']
                    st.rerun()
            st.divider()

def add_to_history(api_desc: str, cases: List[Dict], results: List[Dict]):
    if 'history' not in st.session_state:
        st.session_state.history = []
    metrics = {
        'total': len(cases),
        'passed': len([r for r in results if r['passed']]),
        'failed': len([r for r in results if not r['passed']])
    }
    record = {
        'timestamp': datetime.now().isoformat(),
        'api_desc': api_desc,
        'cases': cases,
        'results': results,
        'metrics': metrics
    }
    st.session_state.history.append(record)

# ========== 关于页面 ==========
def render_about():
    st.markdown("""
    ### 🧪 API TestCraft 智能测试平台
    
    **版本**: v1.0.0  
    **核心技术**:
    - 🤖 基于T5/CodeT5的Seq2Seq生成模型
    - 📐 边界值分析 + 异常值注入的规则引擎
    - 🔄 混合生成策略（模型优先，规则兜底）
    - ⚙️ 可配置的测试执行参数（超时、重试、间隔）
    
    **功能特性**:
    | 功能 | 说明 |
    |------|------|
    | 智能生成 | 从自然语言API描述自动生成测试用例 |
    | 三类覆盖 | 正常值、边界值、异常值全面覆盖 |
    | 即时执行 | 一键执行测试，实时查看结果 |
    | 可视化报告 | 多维度图表展示，支持导出 |
    
    **使用流程**:
    1. 输入API描述（支持自然语言或结构化格式）
    2. 选择要生成的用例类型和数量
    3. 点击生成，AI自动构建测试用例
    4. 配置测试参数（超时、重试等）
    5. 执行测试，查看通过率、响应时间等指标
    6. 导出HTML/JSON报告
    
    ---
    *毕业设计项目 - 基于深度学习的API接口测试用例自动生成方法设计与实现*
    """)

# ========== 主程序 ==========
def main():
    render_header()
    tabs = st.tabs(["📝 生成用例", "▶️ 执行测试", "📊 结果分析", "📚 历史记录", "ℹ️ 关于"])
    
    with tabs[0]:
        api_desc, case_types, num_variants, use_hybrid, generate_btn, uploaded_file = render_api_input()
        
        if generate_btn:
            if not api_desc and uploaded_file is None:
                st.error("⚠️ 请输入API描述或上传OpenAPI文件")
            elif not case_types:
                st.error("⚠️ 请至少选择一种用例类型")
            else:
                with st.spinner("正在生成测试用例..."):
                    cases = generate_cases(api_desc, case_types, num_variants, use_hybrid, uploaded_file)
                    st.session_state.generated_cases = cases
                    st.success(f"✅ 成功生成 {len(cases)} 个测试用例")
        
        render_case_cards(st.session_state.generated_cases)
    
    with tabs[1]:
        if not st.session_state.generated_cases:
            st.warning("👈 请先在'生成用例'标签页创建测试用例")
        else:
            st.markdown(f"### ▶️ 准备执行 {len(st.session_state.generated_cases)} 个测试用例")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                base_url = st.text_input("目标API地址", value="http://localhost:5000", key="base_url")
                timeout = st.slider("请求超时 (秒)", min_value=1, max_value=30, value=10, key="timeout")
            with col2:
                retries = st.slider("失败重试次数", min_value=0, max_value=5, value=0, key="retries")
                delay = st.slider("请求间隔 (秒)", min_value=0.0, max_value=5.0, value=0.0, step=0.1, key="delay")
            with col3:
                parallel = st.toggle("并行执行（实验性）", value=False, key="parallel", disabled=True)
                st.caption("并行功能尚在开发中")
            
            if st.button("🚀 开始执行测试", type="primary", use_container_width=True, key="execute_btn"):
                with st.spinner(""):
                    base_url = st.session_state.get('base_url', 'http://localhost:5000')
                    results = execute_tests(
                        st.session_state.generated_cases,
                        base_url=base_url,
                        timeout=st.session_state.timeout,
                        retries=st.session_state.retries,
                        delay=st.session_state.delay
                    )
                    st.session_state.execution_results = results
                    add_to_history(
                        st.session_state.get('api_desc', 'Unknown'),
                        st.session_state.generated_cases,
                        results
                    )
                    st.success(f"✅ 测试执行完成！通过率: {len([r for r in results if r['passed']])}/{len(results)}")
                    st.balloons()
            
            if st.session_state.execution_results:
                st.info("💡 上次执行结果已加载，切换到'结果分析'标签页查看详情")
    
    with tabs[2]:
        if not st.session_state.execution_results:
            if st.session_state.generated_cases:
                st.warning("👈 请先在'执行测试'标签页运行测试")
            else:
                st.warning("👈 请先生成并执行测试用例")
        else:
            render_results(st.session_state.execution_results)
    
    with tabs[3]:
        render_history()
    
    with tabs[4]:
        render_about()

if __name__ == "__main__":
    main()