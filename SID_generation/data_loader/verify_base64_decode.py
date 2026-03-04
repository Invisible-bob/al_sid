import numpy as np
import base64

def verify_base64_decode(npz_file_path):
    """
    验证NPZ文件中的base64编码是否正确
    """
    print(f"Loading NPZ file: {npz_file_path}")
    
    # 加载NPZ文件
    data = np.load(npz_file_path, allow_pickle=True)
    
    # 获取数据
    item_ids = data['ids']
    emb_features = data['embeds']
    
    print(f"Total items: {len(item_ids)}")
    
    # 验证第一个项目的解码
    if len(item_ids) > 0:
        print(f"\nFirst item:")
        print(f"  Item ID: {item_ids[0]}")
        print(f"  Base64 encoded feature: {emb_features[0]}")
        
        # 解码base64字符串
        try:
            decoded_bytes = base64.b64decode(emb_features[0])
            # 将字节转换回float32数组
            decoded_array = np.frombuffer(decoded_bytes, dtype=np.float32)
            
            print(f"  Decoded array shape: {decoded_array.shape}")
            print(f"  Decoded array dtype: {decoded_array.dtype}")
            print(f"  First 5 elements: {decoded_array[:5]}")
            print(f"  Last 5 elements: {decoded_array[-5:]}")
            
            # 验证维度是否为32
            if len(decoded_array) == 32:
                print("  ✓ Decoded array has correct dimension (32)")
            else:
                print(f"  ✗ Decoded array has incorrect dimension ({len(decoded_array)}, expected 32)")
                
        except Exception as e:
            print(f"  ✗ Error decoding base64: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python verify_base64_decode.py <npz_file_path>")
        sys.exit(1)
    
    npz_file_path = sys.argv[1]
    verify_base64_decode(npz_file_path)
