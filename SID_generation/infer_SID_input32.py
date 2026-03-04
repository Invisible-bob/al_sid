import base64
import csv
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from rqvae_embed.rqvae_clip import RQVAE_EMBED_CLIP

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 全局常量和配置 ---
EXPECTED_EMBEDDING_DIM = 32
CHUNK_SIZE = 5000  # pandas每次读取的行数
BATCH_SIZE = 128  # 模型批次推理的样本数

CKPT_PATH = './result/ae_coin_32_bs2048_lr0.002_ep50_20251113_130733/checkpoint-49.pth'  # todo: 需要推理的ckpt
INPUT_FILE_PATH = './datas/ae_coin_semantic_emb_32.npz'  # 输入的emb，npz格式
OUTPUT_FILE_PATH = './result/ae_coin_32_sid_infer.csv'  # 输出结果

def build_model(ckpt_path: str) -> torch.nn.Module:
    """
    构建并加载预训练的RQ-VAE模型。
    """
    logging.info("开始构建模型...")
    codebook_num = 3
    codebook_size = 8192
    codebook_dim = 64
    input_dim = 32  # 修改为32维

    hps = {
        "bottleneck_type": "rq", "embed_dim": codebook_dim, "n_embed": codebook_size,
        "latent_shape": [8, 8, codebook_dim], "code_shape": [8, 8, codebook_num],
        "shared_codebook": False, "decay": 0.99, "restart_unused_codes": True,
        "loss_type": "cosine", "latent_loss_weight": 0.15, "masked_dropout": 0.0,
        "use_padding_idx": False, "VQ_ema": False, "do_bn": True, 'rotation_trick': False
    }
    ddconfig = {
        "double_z": False, "z_channels": codebook_dim, "resolution": 256, "in_channels": 3,
        "out_ch": 3, "ch": 128, "ch_mult": [1, 1, 2, 2, 4, 4], "num_res_blocks": 2,
        "attn_resolutions": [8], "dropout": 0.00, "input_dim": input_dim
    }

    try:
        model = RQVAE_EMBED_CLIP(hps, ddconfig=ddconfig, checkpointing=True)
        # 确定设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        logging.info(f"正在从 '{ckpt_path}' 加载模型权重...")
        state_dict = torch.load(ckpt_path, map_location=device, weights_only = False)
        model.load_state_dict(state_dict['model'], strict=False)
        logging.info("模型加载成功！")
        return model
    except FileNotFoundError:
        logging.error(f"模型检查点文件未找到: {ckpt_path}")
        raise
    except Exception as e:
        logging.error(f"构建模型时发生未知错误: {e}")
        raise

def predict_batch(
        model: torch.nn.Module,
        item_ids_batch: List[str],
        embeddings_batch: List[np.ndarray]  # 修改类型注解
) -> List[Tuple[str, str]]:
    """
    对一个批次的数据进行解码和模型推理。

    Args:
        model: 已加载的 PyTorch 模型。
        item_ids_batch: 批次的 item_id 列表。
        embeddings_batch: 批次的 32维float32向量列表。

    Returns:
        一个包含 (item_id, SID) 元组的列表。
    """
    valid_item_ids = []
    embedding_list = []

    # 1. 过滤无效数据 (不再需要base64解码)
    for item_id, embedding_np in zip(item_ids_batch, embeddings_batch):
        try:
            # 检查维度是否正确
            if embedding_np.shape[0] != EXPECTED_EMBEDDING_DIM:
                logging.warning(
                    f"Item ID '{item_id}' 的 embedding 维度不正确。 "
                    f"期望维度: {EXPECTED_EMBEDDING_DIM}, 实际维度: {embedding_np.shape[0]}。已跳过此样本。"
                )
                continue  # 跳过这个不符合规范的样本
            embedding_list.append(embedding_np)
            valid_item_ids.append(item_id)
        except Exception as e:
            logging.warning(f"Item ID '{item_id}' 在处理时发生错误: {e}。已跳过。")
            continue

    if not valid_item_ids:
        return []

    # 2. 转换为Tensor并进行推理 (在GPU上执行)
    device = next(model.parameters()).device
    embedding_tensor = torch.from_numpy(np.array(embedding_list)).to(device)

    with torch.no_grad():
        index_batch = model.rq_model.get_codes(embedding_tensor)

    # 3. 将结果转换回CPU并格式化
    cpu_indices = index_batch.cpu().numpy()

    results = []
    for item_id, index_row in zip(valid_item_ids, cpu_indices):
        sid_str = ','.join(index_row.astype(str))
        results.append((item_id, sid_str))

    return results

def process_file(
        model: torch.nn.Module,
        input_path: str,
        output_path: str,
        chunk_size: int,
        batch_size: int
):
    """
    主处理函数，读取NPZ，分批推理，并写入结果。
    """
    try:
        logging.info(f"正在加载NPZ文件 '{input_path}'...")
        data = np.load(input_path)
        ids = data['ids']
        embeddings = data['embeds']
        logging.info(f"NPZ文件加载成功，共有 {len(ids)} 条数据。")
        total_lines = len(ids)
    except FileNotFoundError:
        logging.error(f"输入文件未找到: {input_path}")
        return
    except Exception as e:
        logging.error(f"加载NPZ文件时发生错误: {e}")
        return

    # 使用 with 语句和 csv.writer 确保文件正确关闭和写入
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['item_id', 'SID'])

        # 使用tqdm显示进度
        with tqdm(total=total_lines, desc='Processing data') as pbar:
            try:
                # 分批处理数据
                for i in range(0, total_lines, batch_size):
                    # 获取当前批次的数据
                    ids_batch = ids[i:i+batch_size]
                    embs_batch = embeddings[i:i+batch_size]
                    
                    # 模型推理
                    sid_results = predict_batch(model, ids_batch, embs_batch)

                    # 写入结果
                    if sid_results:
                        writer.writerows(sid_results)

                    pbar.update(len(ids_batch))

            except Exception as e:
                logging.error(f"处理文件时发生未知错误: {e}", exc_info=True)

    logging.info(f"推理完成，结果已保存到: {output_path}")

def main():
    """程序主入口"""
    model = build_model(CKPT_PATH)
    process_file(
        model=model,
        input_path=INPUT_FILE_PATH,
        output_path=OUTPUT_FILE_PATH,
        chunk_size=CHUNK_SIZE,
        batch_size=BATCH_SIZE
    )

if __name__ == "__main__":
    main()