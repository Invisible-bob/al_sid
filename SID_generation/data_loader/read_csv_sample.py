import pandas as pd
import numpy as np
import base64

def decode_embedding(base64_string: str) -> np.ndarray:
    """将一个Base64字符串解码为512维的NumPy向量。"""
    # 从Base64解码，将其解释为float32的缓冲区，然后重塑形状。
    return np.frombuffer(
        base64.b64decode(base64_string),
        dtype=np.float32
    ).reshape(-1)

# 读取CSV文件
file_path = "/home/admin/workspace/aop_lab/repo/al_sid/SID_generation/datas/final_feature/part_0.csv"

# 使用chunksize参数高效读取前10行
chunk_size = 10
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    df = chunk
    break

# 打印列名
print("Column names:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df)

# 解码feature列中的第一个Base64字符串
first_feature = df.iloc[0]['feature']
decoded_vector = decode_embedding(first_feature)

# 打印解码后的向量形状和前10维的值
print(f"\nDecoded vector shape: {decoded_vector.shape}")
print(f"First 10 dimensions of the decoded vector:")
print(decoded_vector[:10])