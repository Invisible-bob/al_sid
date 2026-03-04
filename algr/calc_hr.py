import json
import pandas as pd
import collections
import argparse
import os
from datasets import load_dataset
from tqdm import tqdm
import time

def calculate_hit_rate_k(generate_text, answer, k_list, sid_to_ids_map):
    """
    Calculate HR@K and Validity Rate@K for a single sample
    """
    true_ids = set(map(int, answer.strip().split(";")))
    true_count = len(true_ids)

    hit_count = []
    validity_rates = []

    for k_str in k_list.split(','):
        k = int(k_str)
        # 截取前 k 个生成的 SID
        top_k_sids = generate_text[:k]
        
        generated_ids = []
        valid_sid_count = 0
        
        for sid in top_k_sids:
            if sid in sid_to_ids_map:
                generated_ids.extend(sid_to_ids_map[sid])
                valid_sid_count += 1

        # 计算命中率
        slice_len = min(len(generated_ids), k)
        gids = set(generated_ids[:slice_len])
        hit_count.append(len(gids & true_ids))

        # 计算合法性比率
        total_sid_count = len(top_k_sids)
        validity_rate = valid_sid_count / total_sid_count if total_sid_count > 0 else 0
        validity_rates.append(validity_rate)

    return hit_count, true_count, validity_rates

def calculate_average_hit_rate_k(file_path, k_list, sid_to_ids_map, decoder_only=True):
    """
    Calculate the average HR@K for all samples
    """
    total_count = 0
    hr_count = [0] * len(k_list)
    ohrs = [[] for _ in range(len(k_list))]
    total_validity_rates = [[] for _ in range(len(k_list.split(',')))]  # 存储每个k值下所有样本的SID合法率
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    generate_text = sample["_generated_new_text_"]
                    answer = sample["answer"].split('_')[0]
                    
                    # 修改函数调用以接收合法性统计
                    hit_count, true_count, validity_rates = calculate_hit_rate_k(
                        generate_text, answer, k_list, sid_to_ids_map)
                    
                    total_count += true_count
                    hr_count = [a + b for a, b in zip(hit_count, hr_count)]
                    for i, a in enumerate(hit_count):
                        ohrs[i].append(a / true_count if true_count > 0 else 0)
                    
                    # 收集每个k值下每个样本的SID合法率
                    for i, rate in enumerate(validity_rates):
                        total_validity_rates[i].append(rate)
                    
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON line: {line}")
                    print(f"Error: {e}")
    
    # 计算每个k值下的平均合法率
    avg_validity_rates = []
    for rates in total_validity_rates:
        avg_rate = sum(rates) / len(rates) if rates else 0
        avg_validity_rates.append(avg_rate)
    
    return [sum(ohr) / len(ohr) if len(ohr) > 0 else 0 for ohr in ohrs], avg_validity_rates


def convert_csv_to_map(data):
    """
    Using Pandas column operations + groupby to optimize performance
    Parameters:
        data: Original data (list or DataFrame)
    Returns:
        sid_to_ids_map: Dictionary with keys being SIDs and values ​​being lists of corresponding item_ids
    """
    # 1. Convert to DataFrame and name the columns
    start_time = time.time()
    
    # 1. 如果data是文件路径，直接使用pandas读取（更高效）
    if isinstance(data, str) and data.endswith('.csv'):
        print("Reading CSV file directly with pandas...")
        read_start = time.time()
        df = pd.read_csv(data)
        read_time = time.time() - read_start
        print(f"CSV reading completed in {read_time:.2f} seconds")
    else:
        # 原有的处理逻辑
        print("Converting to DataFrame...")
        conversion_start = time.time()
        df = pd.DataFrame(data).dropna()
        conversion_time = time.time() - conversion_start
        print(f"DataFrame conversion completed in {conversion_time:.2f} seconds")
    
    df.columns = ['item_id', 'codebook_lv1', 'codebook_lv2', 'codebook_lv3']
    
    # 处理空值和NaN值 - 更加完善的版本
    print("Handling missing values...")
    missing_handling_start = time.time()
    # 删除包含空值的行
    df = df.dropna()
    # 删除codebook_lv1, codebook_lv2, codebook_lv3列中包含空字符串的行
    df = df[(df['codebook_lv1'] != '') & (df['codebook_lv2'] != '') & (df['codebook_lv3'] != '')]
    # 确保codebook列是字符串类型
    df['codebook_lv1'] = df['codebook_lv1'].astype(str)
    df['codebook_lv2'] = df['codebook_lv2'].astype(str)
    df['codebook_lv3'] = df['codebook_lv3'].astype(str)
    # 删除非数字值的行，先替换可能的inf值
    import numpy as np
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    # 删除非数字值的行
    numeric_filter = (
        pd.to_numeric(df['codebook_lv1'], errors='coerce').notnull() & 
        pd.to_numeric(df['codebook_lv2'], errors='coerce').notnull() & 
        pd.to_numeric(df['codebook_lv3'], errors='coerce').notnull()
    )
    df = df[numeric_filter]
    missing_handling_time = time.time() - missing_handling_start
    print(f"Missing values handling completed in {missing_handling_time:.2f} seconds")
    print(f"Data shape after cleaning: {df.shape}")

    # 2. Column operation: convert to integer and calculate num2, num3
    print("Processing columns...")
    col_ops_start = time.time()
    # 转换为整数，使用更安全的方式
    df['col1'] = pd.to_numeric(df['codebook_lv1'], errors='coerce').astype('Int64')
    df['col2'] = pd.to_numeric(df['codebook_lv2'], errors='coerce').astype('Int64') + 8192
    df['col3'] = pd.to_numeric(df['codebook_lv3'], errors='coerce').astype('Int64') + 8192 * 2
    # 删除转换后仍为NaN的行
    df = df.dropna()
    # 再次确保是整数类型
    df['col1'] = df['col1'].astype(int)
    df['col2'] = df['col2'].astype(int)
    df['col3'] = df['col3'].astype(int)
    col_ops_time = time.time() - col_ops_start
    print(f"Column operations completed in {col_ops_time:.2f} seconds")

    # 3. Constructing the sid column
    df['sid'] = 'C' + df['col1'].astype(str) + 'C' + df['col2'].astype(str) + 'C' + df['col3'].astype(str)

    # 4. Group by sid and aggregate item_id into a list
    print("Grouping by SID...")
    group_start = time.time()
    sid_to_ids_map = df.groupby('sid')['item_id'].agg(list).to_dict()
    group_time = time.time() - group_start
    print(f"Grouping completed in {group_time:.2f} seconds")

    print("sid_to_ids_map: ", len(sid_to_ids_map))
    return sid_to_ids_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="AL-GR/AL-GR-Tiny")
    parser.add_argument('--item_sid_file', type=str, default="/mnt/nas/nas_fh/yixiang/repo/al_sid/algr/data/item_info/ae_item_sid_lv3_i256_c8192_d64.csv")
    parser.add_argument('--generate_file', type=str, default="/mnt/nas/nas_fh/yixiang/repo/al_sid/algr/logs/generate_qwen2.5_05b_3layer_ae_gr_v3_0/output.jsonl")
    parser.add_argument('--k_list', type=str, default="50,100,200,500,1000")
    parser.add_argument('--decoder_only', action="store_true")
    parser.add_argument('--nebula', action="store_true")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    item_sid_file = args.item_sid_file
    file_path = args.generate_file
    
    ## Write the processed file to the local computer so that it will be faster the next time you run it.
    local_sid2item_file = f"/mnt/nas/nas_fh/yixiang/repo/al_sid/algr/data/sid2item_v_{os.path.basename(item_sid_file).split('.')[0]}.json"
    print("local_sid2item_file: ", local_sid2item_file)
    if os.path.isfile(local_sid2item_file):
        # load JSON file
        print("load file directly")
        with open(local_sid2item_file, "r", encoding="utf-8") as f:
            sid_to_ids_map = json.load(f)
        print("load data success!")
    else:
        print("process data firstly")
        if args.nebula:
            item_sid_data = pd.read_csv(os.path.join(args.dataset_name, args.item_sid_file))
        else:
            item_sid_data = args.item_sid_file  # 传递文件路径而不是加载的数据
        sid_to_ids_map = convert_csv_to_map(item_sid_data)
        # write JSON  file
        with open(local_sid2item_file, "w", encoding="utf-8") as f:
            json.dump(sid_to_ids_map, f, indent=4, ensure_ascii=False)
        print("load data success!")

    # 修改函数调用以接收合法性统计
    result = calculate_average_hit_rate_k(file_path, args.k_list, sid_to_ids_map, args.decoder_only)
    if isinstance(result, tuple) and len(result) == 2:
        average_hr, avg_validity_rates = result
        for k, hr, vr in zip(args.k_list.split(','), average_hr, avg_validity_rates):
            print(f"Average HR@{k}: {hr:.8f}, Average SID Validity Rate@{k}: {vr:.8f}")
    else:
        # 兼容旧版本返回值
        average_hr = result
        for k, hr in zip(args.k_list.split(','), average_hr):
            print(f"Average HR@{k}: {hr:.8f}")

if __name__ == '__main__':
    main()