# src/collect_apis.py
import requests
import json
import os

def download_openapi_specs():
    """下载公开的OpenAPI规范"""
    apis = [
        {
            "name": "petstore",
            "url": "https://petstore.swagger.io/v2/swagger.json",
            "type": "json"
        },
        {
            "name": "github",
            "url": "https://api.github.com/swagger.json",
            "type": "json"
        },
        {
            "name": "uspto",
            "url": "https://developer.uspto.gov/ibd-api/swagger.json",
            "type": "json"
        }
    ]
    
    os.makedirs("data/raw", exist_ok=True)
    
    for api in apis:
        try:
            print(f"下载 {api['name']}...")
            response = requests.get(api['url'], timeout=10)
            if response.status_code == 200:
                if api['type'] == 'json':
                    data = response.json()
                    with open(f"data/raw/{api['name']}.json", "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    print(f"  ✅ 已保存: data/raw/{api['name']}.json")
                else:
                    with open(f"data/raw/{api['name']}.yaml", "w", encoding="utf-8") as f:
                        f.write(response.text)
                    print(f"  ✅ 已保存: data/raw/{api['name']}.yaml")
            else:
                print(f"  ❌ 下载失败: {response.status_code}")
        except Exception as e:
            print(f"  ❌ 错误: {e}")

if __name__ == "__main__":
    download_openapi_specs()