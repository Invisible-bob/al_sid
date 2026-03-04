import csv
import numpy as np
import base64
import argparse
import os
# 添加tqdm用于进度条显示
from tqdm import tqdm
# 添加psutil用于监控内存使用
import psutil
import gc

def convert_csv_to_npz(csv_file_path, npz_file_path, chunk_size=10000):
    """
    将CSV文件转换为NPZ格式
    CSV格式: ids, embeds (emb_features是32维float32，逗号分隔，两端有引号)
    NPZ格式: ids, embeds (emb_features不编码为base64字符串)
    """
    item_ids = []
    emb_features = []
    
    print(f"Reading CSV file: {csv_file_path}")
    
    # 先计算总行数用于进度条
    print("Counting total lines...")
    with open(csv_file_path, 'r') as csvfile:
        total_lines = sum(1 for line in csvfile)
    
    # 重置文件指针并开始处理
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # 使用tqdm显示进度条
        for row_num, row in tqdm(enumerate(reader), total=total_lines, desc="Processing"):
            # 检查行是否有2列
            if len(row) != 2:
                print(f"Warning: Skipping row {row_num} with {len(row)} columns, expected 2")
                continue
                
            try:
                item_id = int(row[0])
                # 解析32维表征字符串（第二列两端有引号，需要去除）
                emb_str = row[1].strip('"')
                emb_values = [float(x) for x in emb_str.split(',')]
                
                # # 确保是32维
                if len(emb_values) != 256:
                    print(f"Warning: Skipping item_id {item_id} with {len(emb_values)} dimensions, expected 256")
                    continue
                    
                item_ids.append(item_id)
                emb_features.append(emb_values)
                
                # 每处理chunk_size行数据后检查内存使用情况
                if len(item_ids) % chunk_size == 0:
                    # 检查内存使用情况
                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent > 90:  # 如果内存使用超过90%
                        print(f"Warning: Memory usage is high ({memory_percent:.1f}%). Consider processing in smaller chunks.")
            except ValueError as e:
                print(f"Warning: Skipping row {row_num} due to value error: {e}")
                continue
    
    # 转换为numpy数组
    print("Converting to numpy arrays...")
    item_ids = np.array(item_ids, dtype=np.int64)
    emb_features = np.array(emb_features, dtype=np.float32)
    
    # 保存为NPZ格式，列名分别为ids和embeds
    print("Saving to NPZ format...")
    np.savez(npz_file_path, ids=item_ids, embeds=emb_features)
    
    print(f"Converted {len(item_ids)} items to NPZ format")
    print(f"Saved to: {npz_file_path}")
    
    # 验证保存的文件
    loaded = np.load(npz_file_path)
    print(f"Verification - ids shape: {loaded['ids'].shape}")
    print(f"Verification - embeds shape: {loaded['embeds'].shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CSV to NPZ format')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output NPZ file path')
    # 添加chunk_size参数
    parser.add_argument('--chunk-size', type=int, default=10000, help='Chunk size for memory management')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        exit(1)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    convert_csv_to_npz(args.input, args.output, args.chunk_size)