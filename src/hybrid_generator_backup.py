import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "models/fine_tuned_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 规则生成器（回退方案）==========
class RuleBasedGenerator:
    """基于规则的测试用例生成器，支持三类用例，针对target_api优化"""
    
    def generate(self, api_desc: str, case_type: str = 'normal') -> dict:
        """根据API描述和用例类型生成测试用例"""
        # 解析HTTP方法和路径模板
        method, path_template = self._parse_method_and_path(api_desc)
        
        # 根据方法确定最终URL
        if method in ['POST']:
            # 创建用户：URL为 /users
            url = '/users'
        elif method in ['GET', 'DELETE']:
            # 如果是获取/删除单个用户，路径中应包含id，如 /users/1
            if '/users' in path_template and '{' in path_template:
                # 生成一个用户ID（假设存在用户ID 1）
                url = '/users/1'
            else:
                # 获取用户列表：URL为 /users
                url = '/users'
        else:
            url = '/users'  # 默认

        # 根据用例类型生成请求体或查询参数
        request = {
            "method": method,
            "url": url,
            "headers": {},
            "query": {},
            "body": None
        }

        # 生成参数值（针对 /users 的 POST 和 GET /users/1）
        if method == 'POST' and url == '/users':
            # 创建用户：需要username, age
            if case_type == 'normal':
                body = {"username": "testuser", "age": 25}
                expected_status = 201  # 成功创建返回201
            elif case_type == 'boundary':
                # 边界值：最小长度/最大长度/最小年龄/最大年龄
                body = {"username": "ab", "age": 0}  # username长度2（小于3，预期失败？但边界值通常取合法边界）
                # 为了测试，我们可以取合法边界：username长度3或20，age 0或120
                # 但这里我们统一让规则生成器生成合法边界值（期望通过）
                body = {"username": "abc", "age": 0}
                expected_status = 201
            else:  # exception
                # 异常值：username太短，年龄负数等
                body = {"username": "a", "age": -5}
                expected_status = 400  # 期望失败
            request['body'] = body
            request['headers']['Content-Type'] = 'application/json'

        elif method == 'GET' and url == '/users':
            # 获取用户列表：无参数
            pass
        elif method == 'GET' and '/users/' in url:
            # 获取单个用户：无参数
            pass
        elif method == 'DELETE' and '/users/' in url:
            # 删除用户：无参数
            pass

        # 默认期望状态码
        expected_status = 200 if case_type != 'exception' else 400
        # 对于POST创建成功返回201
        if method == 'POST' and case_type == 'normal':
            expected_status = 201
        elif method == 'POST' and case_type == 'boundary':
            expected_status = 201  # 合法边界也应该创建成功
        elif method == 'POST' and case_type == 'exception':
            expected_status = 400  # 异常值期望400

        return {
            "description": f"{method} {url} - {case_type} 用例",
            "request": request,
            "expected_response": {"status_code": expected_status},
            "_case_type": case_type
        }

    def _parse_method_and_path(self, desc: str):
        """从描述中提取HTTP方法和路径模板"""
        method = 'GET'
        path = '/users'
        method_match = re.search(r'(GET|POST|PUT|DELETE)', desc, re.IGNORECASE)
        if method_match:
            method = method_match.group(1).upper()
        # 尝试提取类似 /users/{id} 的路径
        path_match = re.search(r'/[a-zA-Z0-9_/{}]+', desc)
        if path_match:
            path = path_match.group()
        return method, path
# ========== 模型生成器 ==========
def load_model():
    print(f"加载模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE)
    model.eval()
    return tokenizer, model

def generate_with_model(desc, tokenizer, model, max_len=256):
    inputs = tokenizer(desc, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_len,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def repair_json(raw):
    """尝试修复模型生成的JSON"""
    raw = raw.strip()
    if not raw.startswith('{'):
        raw = '{' + raw + '}'
    raw = re.sub(r'(\w+)\s*=\s*', r'\1: ', raw)
    raw = re.sub(r'(\b\w+\b)(?=\s*:)', r'"\1"', raw)
    raw = re.sub(r',\s*}', '}', raw)
    return raw

# ========== 混合生成 ==========
def generate_test_case(api_desc, tokenizer, model, rule_gen, case_type='normal'):
    # 先用模型生成
    raw_output = generate_with_model(api_desc, tokenizer, model)
    # 尝试解析
    try:
        case = json.loads(raw_output)
        case['_source'] = 'model'
        return case
    except:
        fixed = repair_json(raw_output)
        try:
            case = json.loads(fixed)
            case['_source'] = 'model_repaired'
            return case
        except:
            # 回退到规则生成，传入 case_type
            case = rule_gen.generate(api_desc, case_type=case_type)
            case['_source'] = 'rule_fallback'
            return case

# ========== 主函数示例 ==========
def main():
    tokenizer, model = load_model()
    rule_gen = RuleBasedGenerator()
    
    test_descriptions = [
        "生成测试用例JSON: POST /users - 创建用户，需要username（字符串，必填，3-20字符）和age（整数，可选，0-120）",
        "生成测试用例JSON: GET /pet/{petId} - 通过ID获取宠物信息，petId为整数",
        "生成测试用例JSON: DELETE /users/{id} - 根据ID删除用户",
    ]
    
    for desc in test_descriptions:
        print("\n" + "="*60)
        print(f"输入: {desc}")
        case = generate_test_case(desc, tokenizer, model, rule_gen)
        print(f"来源: {case.get('_source', 'unknown')}")
        print(json.dumps(case, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()