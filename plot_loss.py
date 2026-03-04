import json
import matplotlib.pyplot as plt
import argparse
import os

# 创建参数解析器
parser = argparse.ArgumentParser(description='Plot training loss from trainer state JSON file.')
parser.add_argument('--input', type=str, required=True, 
                    help='Path to the directory containing trainer_state.json')

args = parser.parse_args()
input_dir = args.input

# 构建文件路径
file_path = os.path.join(input_dir, 'trainer_state.json')

# 读取JSON文件
with open(file_path, 'r') as f:
    data = json.load(f)

# 提取epoch和loss值
epochs = [entry['epoch'] for entry in data['log_history']]
losses = [entry['loss'] for entry in data['log_history']]

# 绘制loss曲线，并减小点的大小
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o', linestyle='-', color='b', markersize=2) 
plt.title('Training Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()

# 保存图像到指定目录，并提高分辨率
output_path = os.path.join(input_dir, 'training_loss_curve.png')
plt.savefig(output_path, dpi=300) 
print(f"Loss curve saved as '{output_path}'")