import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from typing import List, Tuple, Dict, Any, Optional

MODEL_PATH = os.path.abspath("models/fine_tuned_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 模型加载（支持容错）==========
def load_model(model_path: str = None) -> Tuple[Optional[Any], Optional[Any]]:
    """加载微调后的 T5 模型，支持环境变量覆盖路径"""
    path = model_path or os.getenv("TESTGEN_MODEL_PATH", MODEL_PATH)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型路径不存在: {path}。请设置 TESTGEN_MODEL_PATH 环境变量或检查路径。")
    
    print(f"加载模型: {path} (设备: {DEVICE})")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True).to(DEVICE)
    model.eval()
    
    # T5 验证：确保模型是 Seq2Seq 类型
    if not hasattr(model, 'encoder') or not hasattr(model, 'decoder'):
        print("警告: 模型可能不是标准的 Encoder-Decoder 结构")
    
    return tokenizer, model


# ========== 输入解析（增强版）==========
def parse_input(desc: str) -> Tuple[str, str, str, Dict]:
    """
    解析 API 描述，支持更灵活的格式
    返回: (method, path, desc_text, constraints)
    """
    # 移除类型前缀 [NORMAL] 等
    desc_clean = re.sub(r'^\\[[A-Z_]+\\]\\s*', '', desc)
    
    # 支持多种格式：
    # 1. 标准格式: "生成测试用例JSON: POST /users - 描述"
    # 2. 简化格式: "POST /users - 描述"
    # 3. 极简格式: "POST /users 描述"
    
    patterns = [
        r'(?:生成测试用例JSON:)?\\s*(\\w+)\\s+(/[^\\s-]+)\\s*-\\s*(.+)',  # 标准
        r'(?:生成测试用例JSON:)?\\s*(\\w+)\\s+(/[^\\s]+)\\s+(.+)',       # 简化
        r'(?:生成测试用例JSON:)?\\s*(\\w+)\\s+(/[^\\s]+)',                # 只有方法和路径
    ]
    
    method, path, desc_text = "GET", "/unknown", "test"
    
    for pattern in patterns:
        match = re.match(pattern, desc_clean, re.IGNORECASE)
        if match:
            method = match.group(1).upper()
            path = match.group(2)
            desc_text = match.group(3) if len(match.groups()) > 2 else ""
            break
    
    # 提取约束
    constraints = extract_constraints(desc_text)
    
    return method, path, desc_text, constraints


def extract_constraints(desc_text: str) -> Dict:
    """从描述文本中提取字段约束"""
    constraints = {}
    
    # 提取字段名（支持中文括号、英文括号）
    fields = re.findall(r'(\\w+)[（\\(]', desc_text)
    constraints['fields'] = fields
    
    # 长度范围（如 3-20字符、3~20字符）
    length_range = re.search(r'(\\d+)[-~](\\d+)\\s*字符', desc_text)
    if length_range:
        constraints['min_length'] = int(length_range.group(1))
        constraints['max_length'] = int(length_range.group(2))
    else:
        # 单独的最大/最小长度
        max_match = re.search(r'最多(\\d+)字符|不大于(\\d+)字符|(\\d+)字符.*最大', desc_text)
        if max_match:
            constraints['max_length'] = int(max_match.group(1) or max_match.group(2) or max_match.group(3))
        
        min_match = re.search(r'最少(\\d+)字符|不小于(\\d+)字符|(\\d+)字符.*最小', desc_text)
        if min_match:
            constraints['min_length'] = int(min_match.group(1) or min_match.group(2) or min_match.group(3))
    
    # 数值范围（排除字符相关的数字）
    num_range = re.search(r'(\\d+)[-~](\\d+)(?!\\s*字符)(?=.*(?:整数|数字|范围|age|id))', desc_text, re.IGNORECASE)
    if num_range:
        constraints['min_value'] = int(num_range.group(1))
        constraints['max_value'] = int(num_range.group(2))
    
    # 必填/可选字段
    constraints['required_fields'] = re.findall(r'(\\w+)[（\\(][^）)]*必填[^）)]*[）\\)]', desc_text)
    constraints['optional_fields'] = re.findall(r'(\\w+)[（\\(][^）)]*可选[^）)]*[）\\)]', desc_text)
    
    # 字段类型推断
    constraints['field_types'] = {}
    for field in fields:
        if re.search(rf'{field}.*(?:整数|数字|int|age|id)', desc_text, re.IGNORECASE):
            constraints['field_types'][field] = 'integer'
        elif re.search(rf'{field}.*(?:字符串|字符|string|str|username|name)', desc_text, re.IGNORECASE):
            constraints['field_types'][field] = 'string'
        elif re.search(rf'{field}.*(?:布尔|bool)', desc_text, re.IGNORECASE):
            constraints['field_types'][field] = 'boolean'
        else:
            constraints['field_types'][field] = 'string'  # 默认
    
    return constraints


# ========== 模型推理（T5 修复版）==========
def model_inference(
    api_description: str, 
    tokenizer, 
    model, 
    num_sequences: int = 3,
    num_beams: int = 5
) -> Tuple[List[str], List[float]]:
    """
    生成多条候选输出，正确计算 T5 的序列置信度
    """
    inputs = tokenizer(
        api_description,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(DEVICE)
    
    input_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,
            num_beams=num_beams,
            num_return_sequences=num_sequences,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
            temperature=1.0,  # T5-small 建议用默认温度
        )
    
    sequences = outputs.sequences  # [num_sequences, seq_len]
    decoded = [
        tokenizer.decode(seq, skip_special_tokens=True) 
        for seq in sequences
    ]
    
    # 修复：正确计算 T5 的置信度
    confidences = []
    if outputs.scores:
        # outputs.scores 是 tuple，每个元素是 [num_sequences * num_beams, vocab_size]
        # 但我们设置了 num_return_sequences=num_sequences，所以是展平的
        generated_steps = len(outputs.scores)
        
        for i in range(num_sequences):
            token_probs = []
            for step in range(generated_steps):
                # 获取第 i 个候选在第 step 步的分数
                score = outputs.scores[step][i]  # [vocab_size]
                probs = torch.softmax(score, dim=-1)
                
                # 获取实际生成的 token id
                token_id = sequences[i, input_len + step]
                token_prob = probs[token_id].item()
                token_probs.append(token_prob)
            
            # 几何平均（更合理的序列概率）
            if token_probs:
                import math
                log_probs = [math.log(p) for p in token_probs if p > 0]
                avg_log_prob = sum(log_probs) / len(log_probs)
                confidence = round(math.exp(avg_log_prob), 4)
            else:
                confidence = 0.0
            
            confidences.append(confidence)
    else:
        confidences = [0.0] * num_sequences
    
    return decoded, confidences


# ========== 参数提取（增强版）==========
def extract_parameters(text: str) -> Dict:
    """从模型输出中提取参数建议"""
    params = {}
    text_lower = text.lower()
    
    # username 长度（多种模式）
    patterns = [
        r'username.*?length.*?(\\\d+)',
        r'username.*?(\\\d+).*?character',
        r'username.*?(\\\d+).*?字符',
        r'username.*?(\\\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            params['username_len'] = int(match.group(1))
            break
    
    # age 值
    age_patterns = [
        r'age.*?value.*?(\\\d+)',
        r'age.*?(\\\d+).*?year',
        r'age.*?(\\\d+)',
    ]
    for pattern in age_patterns:
        match = re.search(pattern, text_lower)
        if match:
            params['age_value'] = int(match.group(1))
            break
    
    # 状态码（更全面的匹配）
    status_match = re.search(r'\\b(200|201|204|400|401|403|404|422|500|502|503)\\b', text)
    if status_match:
        params['suggested_status'] = int(status_match.group(1))
    
    # 测试类型关键词（用于验证投票结果）
    if any(kw in text_lower for kw in ['boundary', '边界', 'edge', 'limit']):
        params['detected_type_hint'] = 'boundary'
    elif any(kw in text_lower for kw in ['exception', '异常', 'error', 'invalid', 'wrong']):
        params['detected_type_hint'] = 'exception'
    else:
        params['detected_type_hint'] = 'normal'
    
    return params


# ========== 模型输出分析（修复版）==========
def analyze_model_output(
    raw_outputs: List[str], 
    confidences: List[float], 
    constraints: Dict
) -> Dict:
    """
    分析模型输出，投票决定类型，推断字段值
    """
    # 投票决定类型（加权）
    type_votes = {"boundary": 0, "exception": 0, "normal": 0}
    type_keywords = {
        "boundary": ["boundary", "边界", "edge", "limit", "max", "min", "极限"],
        "exception": ["exception", "异常", "error", "invalid", "wrong", "错误", "非法"],
        "normal": ["normal", "正常", "standard", "valid", "正确"]
    }
    
    for i, txt in enumerate(raw_outputs):
        low = txt.lower()
        for ttype, keywords in type_keywords.items():
            if any(kw in low for kw in keywords):
                type_votes[ttype] += confidences[i]  # 按置信度加权
                break
        else:
            type_votes["normal"] += confidences[i] * 0.5  # 无关键词时给 normal 一半权重
    
    case_type = max(type_votes, key=type_votes.get)
    total_votes = sum(type_votes.values())
    vote_confidence = round(type_votes[case_type] / total_votes, 2) if total_votes > 0 else 0.0
    
    # 取置信度最高的候选提取参数
    best_idx = confidences.index(max(confidences))
    best_output = raw_outputs[best_idx]
    extracted_params = extract_parameters(best_output)
    
    # 验证投票结果与提取的参数是否一致
    if extracted_params.get('detected_type_hint') != case_type:
        print(f"   [警告] 投票类型({case_type})与参数提取类型({extracted_params.get('detected_type_hint')})不一致")
    
    # 根据类型和约束推断字段值（修复版）
    field_suggestions = generate_field_values(case_type, constraints, extracted_params)
    
    # 综合置信度
    avg_model_conf = round(sum(confidences) / len(confidences), 4)
    overall_confidence = round(vote_confidence * 0.4 + avg_model_conf * 0.6, 4)
    
    return {
        "case_type": case_type,
        "type_votes": type_votes,
        "vote_confidence": vote_confidence,
        "model_confidence": avg_model_conf,
        "overall_confidence": overall_confidence,
        "extracted_params": extracted_params,
        "field_suggestions": field_suggestions,
        "best_raw_output": best_output,
        "best_idx": best_idx,
        "all_candidates": raw_outputs,
        "all_confidences": confidences
    }


def generate_field_values(case_type: str, constraints: Dict, extracted: Dict) -> Dict:
    """根据测试类型生成字段值（修复边界值和异常值）"""
    field_suggestions = {}
    
    max_len = constraints.get('max_length', 20)
    min_len = constraints.get('min_length', 3)
    max_val = constraints.get('max_value', 120)
    min_val = constraints.get('min_value', 0)
    field_types = constraints.get('field_types', {})
    
    for field in constraints.get('fields', []):
        field_type = field_types.get(field, 'string')
        
        if case_type == "normal":
            # 正常值：中间值
            if field_type == 'integer':
                field_suggestions[field] = (max_val + min_val) // 2
            elif field_type == 'string':
                mid_len = (max_len + min_len) // 2
                field_suggestions[field] = "t" * max(mid_len, 1)
            elif field_type == 'boolean':
                field_suggestions[field] = True
                
        elif case_type == "boundary":
            # 修复：边界值同时测试最小和最大
            if field_type == 'integer':
                # 根据字段名或随机选择测试哪个边界
                import hashlib
                hash_val = int(hashlib.md5(field.encode()).hexdigest(), 16)
                if hash_val % 2 == 0:
                    field_suggestions[field] = max_val  # 最大边界
                else:
                    field_suggestions[field] = min_val  # 最小边界
            elif field_type == 'string':
                hash_val = int(hashlib.md5(field.encode()).hexdigest(), 16)
                if hash_val % 2 == 0:
                    field_suggestions[field] = "a" * max_len  # 最大长度
                else:
                    field_suggestions[field] = "a" * min_len  # 最小长度
                    
        else:  # exception
            # 修复：根据字段类型推断合理的异常值
            if field_type == 'integer':
                # 整数字段的异常值
                exceptions = ["not_a_number", None, -99999, max_val + 1000]
                import hashlib
                hash_val = int(hashlib.md5(field.encode()).hexdigest(), 16)
                field_suggestions[field] = exceptions[hash_val % len(exceptions)]
            elif field_type == 'string':
                # 字符串字段的异常值
                exceptions = [
                    None,  # null
                    "",    # 空字符串
                    "a" * (max_len + 100),  # 超长
                    "<script>alert(1)</script>",  # XSS 尝试
                    "\\u0000",  # 空字符
                ]
                import hashlib
                hash_val = int(hashlib.md5(field.encode()).hexdigest(), 16)
                field_suggestions[field] = exceptions[hash_val % len(exceptions)]
            elif field_type == 'boolean':
                field_suggestions[field] = "not_boolean"
    
    return field_suggestions


# ========== 模板组装（修复版）==========
def assemble_test_case(
    method: str, 
    path: str, 
    desc_text: str, 
    analysis: Dict, 
    constraints: Dict
) -> Dict:
    """使用模型分析结果构建最终测试用例 JSON"""
    case_type = analysis["case_type"]
    field_suggestions = analysis["field_suggestions"]
    extracted = analysis["extracted_params"]
    
    # 修复：路径参数使用更合理的异常值
    path_params = re.findall(r'\\{(\\w+)\\}', path)
    url = path
    for param in path_params:
        if case_type == "normal":
            val = 1  # 正常值
        elif case_type == "boundary":
            # 边界：极大值
            val = 999999
        else:
            # 异常：负数或字符串（更合理）
            val = "-1"
        url = url.replace(f'\\u007b{param}\\u007d', str(val))
    
    # 请求体
    body = {}
    if method in ["POST", "PUT", "PATCH"]:
        if field_suggestions:
            body = dict(field_suggestions)
        else:
            # 回退到基于规则的默认值
            body = generate_fallback_body(case_type, extracted, constraints)
    
    # 期望状态码（优先使用模型建议）
    expected_status = infer_status_code(method, case_type, path_params, extracted)
    
    # 组装 JSON
    test_case = {
        "description": f"{method} {path} - {case_type} test",
        "case_type": case_type,
        "confidence": analysis["overall_confidence"],
        "model_analysis": {
            "case_type_votes": analysis["type_votes"],
            "vote_confidence": analysis["vote_confidence"],
            "model_confidence": analysis["model_confidence"],
            "overall_confidence": analysis["overall_confidence"],
            "extracted_params": extracted,
            "field_suggestions": field_suggestions,
            "best_raw_output": analysis["best_raw_output"][:120],
            "best_candidate_idx": analysis["best_idx"]
        },
        "request": {
            "method": method,
            "url": url,
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            "query": {},
            "body": body
        },
        "expected_response": {
            "status_code": expected_status,
            "headers": {"Content-Type": "application/json"}
        }
    }
    
    # 响应体示例
    if method == "GET" and case_type == "normal" and expected_status == 200:
        test_case["expected_response"]["body"] = {
            "id": 1, 
            "username": "preloaded", 
            "age": 30
        }
    elif case_type != "normal":
        test_case["expected_response"]["body"] = {
            "error": "Invalid request",
            "code": "VALIDATION_ERROR",
            "details": f"Failed validation for {case_type} case"
        }
    
    return test_case


def generate_fallback_body(case_type: str, extracted: Dict, constraints: Dict) -> Dict:
    """当模型无法推断字段时使用的回退逻辑"""
    max_len = extracted.get('username_len', 100)
    age_val = extracted.get('age_value', 999999)
    
    if case_type == "normal":
        return {"username": "testuser", "age": 25}
    elif case_type == "boundary":
        return {"username": "a" * max_len, "age": age_val}
    else:
        return {"username": None, "age": "not_a_number"}


def infer_status_code(method: str, case_type: str, path_params: List[str], extracted: Dict) -> int:
    """推断期望的状态码"""
    if "suggested_status" in extracted:
        return extracted["suggested_status"]
    
    if case_type == "normal":
        if method == "POST":
            return 201
        elif method == "DELETE":
            return 204 if path_params else 200
        else:
            return 200
    else:  # boundary 或 exception
        if method == "POST":
            return 400  # Bad Request
        elif method in ["GET", "DELETE"] and path_params:
            return 404 if case_type == "exception" else 400
        else:
            return 400


# ========== 主生成函数（对外接口）==========
def generate_test_case(api_description: str, tokenizer, model) -> Dict:
    """
    对外接口：输入API描述，返回测试用例 JSON
    """
    # 1. 解析输入
    method, path, desc_text, constraints = parse_input(api_description)
    print(f"\\n   [解析] 方法={method}, 路径={path}")
    print(f"   [解析] 约束={constraints}")
    
    # 2. 模型推理（生成3条候选）
    raw_outputs, confidences = model_inference(api_description, tokenizer, model, num_sequences=3)
    print(f"   [模型] 候选输出:")
    for i, (txt, conf) in enumerate(zip(raw_outputs, confidences)):
        print(f"          候选{i+1} (置信度={conf:.4f}): {txt[:80]}")
    
    # 3. 分析输出
    analysis = analyze_model_output(raw_outputs, confidences, constraints)
    print(f"   [分析] 类型投票={analysis['type_votes']}")
    print(f"   [分析] 选中类型={analysis['case_type']} (置信度={analysis['overall_confidence']})")
    print(f"   [分析] 字段推断={analysis['field_suggestions']}")
    
    # 4. 组装最终用例
    return assemble_test_case(method, path, desc_text, analysis, constraints)


# ========== 批量生成（强制三类用例）==========
def generate_all_cases(api_description: str, tokenizer, model) -> List[Dict]:
    """
    为同一 API 强制生成 normal / boundary / exception 三类用例
    """
    type_prefixes = {
        "normal":    "生成正常测试用例JSON: ",
        "boundary":  "生成边界测试用例JSON: ",
        "exception": "生成异常测试用例JSON: ",
    }
    
    # 提取纯 API 描述（移除原始前缀）
    pure_desc = re.sub(r'^生成测试用例JSON:\\s*', '', api_description, flags=re.IGNORECASE)
    
    results = []
    for ctype, prefix in type_prefixes.items():
        prompted = prefix + pure_desc
        try:
            result = generate_test_case(prompted, tokenizer, model)
            result["forced_type"] = ctype
            result["generation_prompt"] = prompted
            results.append(result)
            print(f"   ✅ 生成 {ctype} 用例完成 (置信度: {result['confidence']})")
        except Exception as e:
            print(f"   ❌ 生成 {ctype} 用例失败: {e}")
            # 生成一个带错误信息的占位用例
            results.append({
                "forced_type": ctype,
                "error": str(e),
                "description": f"Failed to generate {ctype} test case"
            })
    
    return results


# ========== 测试入口 ==========
if __name__ == "__main__":
    # 加载模型
    tokenizer, model = load_model()
    
    # 测试用例
    test_descriptions = [
        "生成测试用例JSON: POST /users - 创建用户，需要username（字符串，必填，3-20字符）和age（整数，可选，0-120）",
        "生成测试用例JSON: GET /users/{id} - 获取用户信息，id为整数",
        "生成测试用例JSON: DELETE /users/{id} - 删除用户，id为整数",
        "POST /products - 创建商品，name（字符串，必填，1-100字符），price（数字，必填，0.01-999999）",
    ]
    
    for desc in test_descriptions:
        print("\\n" + "=" * 70)
        print(f"输入: {desc}")
        print("=" * 70)
        
        # 方式A：单次生成（自动判断类型）
        try:
            single = generate_test_case(desc, tokenizer, model)
            print("\\n── 单次生成结果 ──")
            print(json.dumps(single, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"\\n单次生成失败: {e}")
        
        # 方式B：批量生成三种类型
        print("\\n── 批量生成（三类用例）──")
        all_cases = generate_all_cases(desc, tokenizer, model)
        for case in all_cases:
            if "error" not in case:
                print(f"\\n--- {case.get('forced_type', 'unknown')} ---")
                print(json.dumps(case, indent=2, ensure_ascii=False))