import json
from datetime import datetime

def create_example_dataset():
    example_data = [
        # ========== 用户管理 API ==========
        {
            "api_description": "POST /users - 创建用户，需要username（字符串，必填，3-20字符）和age（整数，可选，0-120）",
            "endpoint": "/users",
            "method": "POST",
            "case_type": "normal",
            "test_case": {
                "description": "创建用户 - 正常值",
                "request": {
                    "method": "POST",
                    "url": "http://api.example.com/users",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"username": "john_doe", "age": 25}
                },
                "expected_response": {"status_code": 201}
            }
        },
        {
            "api_description": "POST /users - 创建用户，需要username（字符串，必填，3-20字符）和age（整数，可选，0-120）",
            "endpoint": "/users",
            "method": "POST",
            "case_type": "boundary",
            "test_case": {
                "description": "创建用户 - 边界值（最小年龄0）",
                "request": {
                    "method": "POST",
                    "url": "http://api.example.com/users",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"username": "alice", "age": 0}
                },
                "expected_response": {"status_code": 201}
            }
        },
        {
            "api_description": "POST /users - 创建用户，需要username（字符串，必填，3-20字符）和age（整数，可选，0-120）",
            "endpoint": "/users",
            "method": "POST",
            "case_type": "boundary",
            "test_case": {
                "description": "创建用户 - 边界值（最大年龄120）",
                "request": {
                    "method": "POST",
                    "url": "http://api.example.com/users",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"username": "bob", "age": 120}
                },
                "expected_response": {"status_code": 201}
            }
        },
        {
            "api_description": "POST /users - 创建用户，需要username（字符串，必填，3-20字符）和age（整数，可选，0-120）",
            "endpoint": "/users",
            "method": "POST",
            "case_type": "exception",
            "test_case": {
                "description": "创建用户 - 异常值（年龄负数）",
                "request": {
                    "method": "POST",
                    "url": "http://api.example.com/users",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"username": "charlie", "age": -5}
                },
                "expected_response": {"status_code": 400}
            }
        },
        {
            "api_description": "POST /users - 创建用户，需要username（字符串，必填，3-20字符）和age（整数，可选，0-120）",
            "endpoint": "/users",
            "method": "POST",
            "case_type": "exception",
            "test_case": {
                "description": "创建用户 - 异常值（年龄超过120）",
                "request": {
                    "method": "POST",
                    "url": "http://api.example.com/users",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"username": "david", "age": 150}
                },
                "expected_response": {"status_code": 400}
            }
        },
        {
            "api_description": "GET /users - 获取用户列表，支持分页参数page（整数，可选）和limit（整数，可选，1-100）",
            "endpoint": "/users",
            "method": "GET",
            "case_type": "normal",
            "test_case": {
                "description": "获取用户列表 - 正常值",
                "request": {
                    "method": "GET",
                    "url": "http://api.example.com/users",
                    "headers": {},
                    "query": {"page": 1, "limit": 20}
                },
                "expected_response": {"status_code": 200}
            }
        },
        {
            "api_description": "GET /users/{id} - 根据ID获取用户信息，id为整数路径参数",
            "endpoint": "/users/{id}",
            "method": "GET",
            "case_type": "normal",
            "test_case": {
                "description": "获取用户详情 - 正常值",
                "request": {
                    "method": "GET",
                    "url": "http://api.example.com/users/42",
                    "headers": {},
                    "body": {}
                },
                "expected_response": {"status_code": 200}
            }
        },
        {
            "api_description": "PUT /users/{id} - 更新用户信息，需要username（字符串，可选）和age（整数，可选）",
            "endpoint": "/users/{id}",
            "method": "PUT",
            "case_type": "normal",
            "test_case": {
                "description": "更新用户 - 正常值",
                "request": {
                    "method": "PUT",
                    "url": "http://api.example.com/users/42",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"username": "updated_name", "age": 30}
                },
                "expected_response": {"status_code": 200}
            }
        },
        {
            "api_description": "DELETE /users/{id} - 删除用户，id为整数路径参数",
            "endpoint": "/users/{id}",
            "method": "DELETE",
            "case_type": "normal",
            "test_case": {
                "description": "删除用户 - 正常值",
                "request": {
                    "method": "DELETE",
                    "url": "http://api.example.com/users/42",
                    "headers": {},
                    "body": {}
                },
                "expected_response": {"status_code": 204}
            }
        },
        
        # ========== 宠物商店 API ==========
        {
            "api_description": "GET /pet/{petId} - 通过ID获取宠物信息，petId为整数",
            "endpoint": "/pet/{petId}",
            "method": "GET",
            "case_type": "normal",
            "test_case": {
                "description": "获取宠物信息 - 正常值",
                "request": {
                    "method": "GET",
                    "url": "http://petstore.example.com/pet/123",
                    "headers": {},
                    "body": {}
                },
                "expected_response": {"status_code": 200}
            }
        },
        {
            "api_description": "GET /pet/{petId} - 通过ID获取宠物信息，petId为整数",
            "endpoint": "/pet/{petId}",
            "method": "GET",
            "case_type": "exception",
            "test_case": {
                "description": "获取宠物信息 - 异常值（负数ID）",
                "request": {
                    "method": "GET",
                    "url": "http://petstore.example.com/pet/-1",
                    "headers": {},
                    "body": {}
                },
                "expected_response": {"status_code": 404}
            }
        },
        {
            "api_description": "POST /pet - 添加新宠物，需要name（字符串，必填）、status（字符串，可选，枚举值：available/pending/sold）",
            "endpoint": "/pet",
            "method": "POST",
            "case_type": "normal",
            "test_case": {
                "description": "添加宠物 - 正常值",
                "request": {
                    "method": "POST",
                    "url": "http://petstore.example.com/pet",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"name": "doggie", "status": "available"}
                },
                "expected_response": {"status_code": 200}
            }
        },
        
        # ========== 商品库存 API ==========
        {
            "api_description": "PUT /products/{id}/stock - 更新商品库存，需要quantity（整数，必填，1-1000）",
            "endpoint": "/products/{id}/stock",
            "method": "PUT",
            "case_type": "normal",
            "test_case": {
                "description": "更新库存 - 正常值",
                "request": {
                    "method": "PUT",
                    "url": "http://api.example.com/products/42/stock",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"quantity": 500}
                },
                "expected_response": {"status_code": 200}
            }
        },
        {
            "api_description": "PUT /products/{id}/stock - 更新商品库存，需要quantity（整数，必填，1-1000）",
            "endpoint": "/products/{id}/stock",
            "method": "PUT",
            "case_type": "boundary",
            "test_case": {
                "description": "更新库存 - 边界值（最小库存1）",
                "request": {
                    "method": "PUT",
                    "url": "http://api.example.com/products/42/stock",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"quantity": 1}
                },
                "expected_response": {"status_code": 200}
            }
        },
        {
            "api_description": "PUT /products/{id}/stock - 更新商品库存，需要quantity（整数，必填，1-1000）",
            "endpoint": "/products/{id}/stock",
            "method": "PUT",
            "case_type": "boundary",
            "test_case": {
                "description": "更新库存 - 边界值（最大库存1000）",
                "request": {
                    "method": "PUT",
                    "url": "http://api.example.com/products/42/stock",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"quantity": 1000}
                },
                "expected_response": {"status_code": 200}
            }
        },
        {
            "api_description": "PUT /products/{id}/stock - 更新商品库存，需要quantity（整数，必填，1-1000）",
            "endpoint": "/products/{id}/stock",
            "method": "PUT",
            "case_type": "exception",
            "test_case": {
                "description": "更新库存 - 异常值（库存为0）",
                "request": {
                    "method": "PUT",
                    "url": "http://api.example.com/products/42/stock",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"quantity": 0}
                },
                "expected_response": {"status_code": 400}
            }
        },
        {
            "api_description": "PUT /products/{id}/stock - 更新商品库存，需要quantity（整数，必填，1-1000）",
            "endpoint": "/products/{id}/stock",
            "method": "PUT",
            "case_type": "exception",
            "test_case": {
                "description": "更新库存 - 异常值（库存为1001）",
                "request": {
                    "method": "PUT",
                    "url": "http://api.example.com/products/42/stock",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"quantity": 1001}
                },
                "expected_response": {"status_code": 400}
            }
        },
        {
            "api_description": "GET /products/{id} - 查询商品信息，返回商品详情",
            "endpoint": "/products/{id}",
            "method": "GET",
            "case_type": "normal",
            "test_case": {
                "description": "查询商品 - 正常值",
                "request": {
                    "method": "GET",
                    "url": "http://api.example.com/products/42",
                    "headers": {},
                    "body": {}
                },
                "expected_response": {"status_code": 200}
            }
        },
    ]
    
    final_data = []
    for i, item in enumerate(example_data, 1):
        final_item = {
            "id": f"example_{i:03d}",
            "created_at": datetime.now().isoformat(),
            **item
        }
        final_data.append(final_item)
    
    import os
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "example_dataset.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已创建示例数据集: {output_file}")
    print(f"   包含 {len(final_data)} 个测试用例")
    
    # 统计
    types = {}
    for item in final_data:
        t = item['case_type']
        types[t] = types.get(t, 0) + 1
    print(f"   类型分布: {types}")
    
    return final_data

if __name__ == "__main__":
    create_example_dataset()