import numpy as np
import sys

def check_npz_content(npz_file_path):
    """
    检查NPZ文件内容
    """
    print(f"Loading NPZ file: {npz_file_path}")
    
    # 加载NPZ文件
    data = np.load(npz_file_path, allow_pickle=True)
    
    # 显示文件中的键
    print("Keys in NPZ file:", list(data.keys()))
    
    # 显示item_id数据
    item_ids = data['ids']
    print(f"\nItem IDs (shape: {item_ids.shape}):")
    print(item_ids)
    
    # 显示emb_features数据
    emb_features = data['embeds']
    print(f"\nEmbedded features (shape: {emb_features.shape}):")
    print(emb_features)
    
    # 显示一些统计信息
    print(f"\nTotal items: {len(item_ids)}")
    
    # 显示第一个项目的详细信息
    if len(item_ids) > 0:
        print(f"\nFirst item:")
        print(f"  Item ID: {item_ids[0]}")
        print(f"  Embedded feature: {emb_features[0]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_npz_content.py <npz_file_path>")
        sys.exit(1)
    
    npz_file_path = sys.argv[1]
    check_npz_content(npz_file_path)
