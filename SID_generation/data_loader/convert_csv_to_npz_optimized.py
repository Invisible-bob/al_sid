import csv
import numpy as np
import argparse
import os
import time
from multiprocessing import Pool, cpu_count
import tempfile
import gc
import psutil
from tqdm import tqdm

def process_chunk_file(chunk_file_info):
    """处理数据块文件"""
    chunk_file_path, expected_dim = chunk_file_info
    item_ids = []
    emb_features = []
    
    with open(chunk_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                # 检查行是否有2列
                if len(row) != 2:
                    continue
                    
                item_id = int(row[0])
                # 解析表征字符串（第二列两端有引号，需要去除）
                emb_str = row[1].strip('"')
                emb_values = [float(x) for x in emb_str.split(',')]
                
                # 确保是expected_dim维
                if len(emb_values) != expected_dim:
                    continue
                    
                item_ids.append(item_id)
                emb_features.append(emb_values)
            except (ValueError, IndexError) as e:
                continue
    
    return item_ids, emb_features

def split_csv_file(csv_file_path, lines_per_chunk=50000):
    """将大CSV文件分割成小块"""
    chunk_files = []
    chunk_idx = 0
    line_count = 0
    
    temp_dir = tempfile.mkdtemp()
    current_chunk_file = None
    current_chunk_writer = None
    
    with open(csv_file_path, 'r') as infile:
        for line in infile:
            if line_count % lines_per_chunk == 0:
                # 关闭当前块文件
                if current_chunk_file:
                    current_chunk_file.close()
                
                # 创建新块文件
                chunk_file_path = os.path.join(temp_dir, f"chunk_{chunk_idx}.csv")
                chunk_files.append(chunk_file_path)
                current_chunk_file = open(chunk_file_path, 'w')
                chunk_idx += 1
            
            current_chunk_file.write(line)
            line_count += 1
        
        # 关闭最后一个块文件
        if current_chunk_file:
            current_chunk_file.close()
    
    return chunk_files, temp_dir

def monitor_resources():
    """监控系统资源使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    system_memory = psutil.virtual_memory()
    
    return {
        'process_memory_mb': memory_info.rss / 1024 / 1024,
        'system_memory_percent': system_memory.percent,
        'cpu_percent': process.cpu_percent()
    }

def convert_csv_to_npz_optimized(csv_file_path, npz_file_path, chunk_size=50000, n_processes=None, expected_dim=256, 
                               memory_limit_gb=8):
    """
    优化版本：将CSV文件转换为NPZ格式，使用多进程并行处理提升效率
    """
    start_time = time.time()
    
    if n_processes is None:
        n_processes = min(cpu_count(), 32)  # 限制最大进程数为32
    
    print(f"Reading CSV file: {csv_file_path}")
    print(f"Using {n_processes} processes for parallel processing")
    print(f"Chunk size: {chunk_size}")
    print(f"Expected embedding dimension: {expected_dim}")
    
    # 监控初始资源使用
    initial_resources = monitor_resources()
    print(f"Initial memory usage: {initial_resources['process_memory_mb']:.2f} MB")
    
    # 检查输入文件大小
    file_size_gb = os.path.getsize(csv_file_path) / (1024**3)
    print(f"Input file size: {file_size_gb:.2f} GB")
    
    # 计算总行数用于进度条
    print("Counting total lines for progress tracking...")
    total_lines = 0
    with open(csv_file_path, 'r') as f:
        for _ in tqdm(f, desc="Counting lines", unit="lines"):
            total_lines += 1
    
    print(f"Total lines: {total_lines}")
    
    # 将大CSV文件分割成小块
    print("Splitting CSV file into chunks...")
    chunk_files, temp_dir = split_csv_file(csv_file_path, chunk_size)
    print(f"Split into {len(chunk_files)} chunks")
    
    # 准备处理参数
    process_params = [(chunk_file, expected_dim) for chunk_file in chunk_files]
    
    # 使用多进程处理所有块，添加进度条
    print("Processing chunks in parallel...")
    pool = Pool(processes=n_processes)
    
    try:
        # 并行处理所有块，显示进度条
        results = []
        with tqdm(total=len(process_params), desc="Processing chunks", unit="chunks") as pbar:
            # 使用imap_unordered以提高效率
            for result in pool.imap_unordered(process_chunk_file, process_params):
                results.append(result)
                pbar.update(1)
        
        pool.close()
        pool.join()
        
        # 合并所有结果
        print("Merging results...")
        all_item_ids = []
        all_emb_features = []
        
        # 合并时也显示进度条
        with tqdm(total=len(results), desc="Merging results", unit="chunks") as pbar:
            for item_ids, emb_features in results:
                if item_ids:  # 只合并非空数据
                    all_item_ids.extend(item_ids)
                    all_emb_features.extend(emb_features)
                pbar.update(1)
        
        # 转换为numpy数组
        print("Converting to numpy arrays...")
        final_ids = np.array(all_item_ids, dtype=np.int64)
        final_embeds = np.array(all_emb_features, dtype=np.float32)
        
        # 保存为NPZ格式
        print("Saving to NPZ format...")
        save_start_time = time.time()
        np.savez(npz_file_path, ids=final_ids, embeds=final_embeds)
        save_time = time.time() - save_start_time
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Converted {len(final_ids)} items to NPZ format")
        print(f"Processing time: {total_time:.2f}s")
        if total_time > 0:
            print(f"Performance: {len(final_ids)/total_time:.2f} records/second")
        print(f"Saving time: {save_time:.2f}s")
        print(f"Saved to: {npz_file_path}")
        
        # 验证保存的文件
        try:
            loaded = np.load(npz_file_path)
            print(f"Verification - ids shape: {loaded['ids'].shape}")
            print(f"Verification - embeds shape: {loaded['embeds'].shape}")
        except Exception as e:
            print(f"Warning: Failed to verify output file: {e}")
        
        # 最终资源使用情况
        final_resources = monitor_resources()
        print(f"Final memory usage: {final_resources['process_memory_mb']:.2f} MB")
        print(f"Memory growth: {final_resources['process_memory_mb'] - initial_resources['process_memory_mb']:.2f} MB")
        
    finally:
        # 清理临时文件和目录
        pool.terminate()
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                try:
                    os.remove(chunk_file)
                except Exception as e:
                    print(f"Warning: Failed to remove temporary file {chunk_file}: {e}")
        try:
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to remove temporary directory {temp_dir}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CSV to NPZ format (Optimized with Multiprocessing)')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output NPZ file path')
    parser.add_argument('--chunk-size', type=int, default=50000, help='Chunk size for processing (default: 50000)')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes for parallel processing (default: CPU cores, max 32)')
    parser.add_argument('--dim', type=int, default=256, help='Expected embedding dimension (default: 256)')
    parser.add_argument('--memory-limit', type=int, default=128, help='Memory limit in GB (default: 128)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        exit(1)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    convert_csv_to_npz_optimized(args.input, args.output, args.chunk_size, args.processes, args.dim, 
                                args.memory_limit)