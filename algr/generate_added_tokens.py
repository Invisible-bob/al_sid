import json

def generate_added_tokens_json():
    """
    生成 added_tokens.json 文件，包含所有新增的 token
    包括：[SEP] 特殊 token 和 C0-C65535 普通 token
    """
    added_tokens = []
    
    # 添加 [SEP] 特殊 token
    added_tokens.append({
        "content": "[SEP]",
        "single_word": True,
        "lstrip": True,
        "rstrip": True,
        "special": True  # [SEP] 是特殊 token
    })
    
    # 添加 C0 到 C65535 的普通 token
    for i in range(0, 3 * 8192):  # 0 到 65535
        added_tokens.append({
            "content": f"C{i}",
            "single_word": True,
            "lstrip": True,
            "rstrip": True,
            "special": False  # C tokens 是普通 token
        })
    
    # 写入 JSON 文件
    with open('added_tokens.json', 'w', encoding='utf-8') as f:
        json.dump(added_tokens, f, indent=2, ensure_ascii=False)
    
    print(f"成功生成 added_tokens.json，共 {len(added_tokens)} 个 token")

if __name__ == "__main__":
    generate_added_tokens_json()
