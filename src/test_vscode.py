"""
VS Code 配置测试文件
"""

def test_environment():
    """测试环境配置"""
    import sys
    import os
    
    print("=" * 60)
    print("VS Code 工作区配置测试")
    print("=" * 60)
    
    # 1. 检查Python解释器
    print(f"1. Python解释器: {sys.executable}")
    
    # 2. 检查是否在虚拟环境中
    in_venv = 'venv' in sys.executable
    print(f"2. 虚拟环境: {'✅ 已激活' if in_venv else '❌ 未激活'}")
    
    # 3. 检查工作目录
    print(f"3. 工作目录: {os.getcwd()}")
    
    # 4. 测试导入库
    print("4. 库导入测试:")
    libraries = ['torch', 'transformers', 'requests', 'numpy', 'pandas']
    
    for lib in libraries:
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', '未知版本')
            print(f"   ✅ {lib}: {version}")
        except ImportError:
            print(f"   ❌ {lib}: 未安装")
    
    # 5. 测试文件系统访问
    print("5. 文件系统测试:")
    folders = ['data', 'src', 'models', 'tests']
    for folder in folders:
        if os.path.exists(folder):
            print(f"   ✅ {folder}: 存在")
        else:
            print(f"   ❌ {folder}: 不存在")
    
    print("=" * 60)
    
    if in_venv:
        print("🎉 VS Code 配置成功！可以开始开发了。")
    else:
        print("⚠️  请确保选择了正确的Python解释器。")
    
    return in_venv

if __name__ == "__main__":
    test_environment()