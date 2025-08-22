import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from FusionModel import translator,TQLayer


seq_length = 6  # 预测步长
n_layers=4
n_qubits=6




# 设置随机种子以确保可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
set_seed()

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 加载 Santa Fe 时间序列数据
def get_santa_fe_data():
    time_series_raw = np.load("./sk_Santa_Fe_2000.npy")
    K = len(time_series_raw)
    # normalize data
    min_ts = min(time_series_raw)
    max_ts = max(time_series_raw)
    time_series = (time_series_raw + np.abs(min_ts)) / (max_ts - min_ts)
    # flatten time series
    time_series = time_series.flatten()
    return time_series, min_ts, max_ts


# 准备时间序列数据
def prepare_time_series_data(data, seq_length):
    """
    将时间序列转换为监督学习数据集
    :param time_series: 时间序列数据
    :param seq_length: 输入序列长度
    :param forecast_horizon: 预测步长
    :return: (X, y) 输入输出对
    """
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)


# 加载和准备数据
time_series, min_ts, max_ts = get_santa_fe_data()
X, y = prepare_time_series_data(time_series, seq_length)

# 划分训练集和测试集
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)  # 添加特征维度
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test).unsqueeze(-1)
y_test_tensor = torch.FloatTensor(y_test)

# 创建数据加载器
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 量子模型定义

single = [[i] + [1] * 2 * n_layers for i in range(1, n_qubits + 1)]
enta = [[i] + [i + 1] * n_layers for i in range(1, n_qubits)] + [[n_qubits] + [1] * n_layers]

design = translator(single, enta, 'full', (n_qubits, n_layers), 1)
model=TQLayer(n_qubits, design,6,1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)


# 计算归一化均方误差 (NMSE)
def nmse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2) / np.var(y_true)


# 训练函数
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    progress_bar = tqdm(train_loader, desc='Training', unit='batch')
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()

        # 在进度条中实时显示当前损失
        progress_bar.set_postfix(loss=f'{loss.item():.6f}')
        all_preds.append(output.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

    # 计算整个训练集的NMSE
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    train_nmse = nmse(all_targets, all_preds)

    return running_loss / len(train_loader), train_nmse


# 测试函数
def test_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            all_preds.append(output.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())

    # 计算整个测试集的NMSE
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    test_nmse = nmse(all_targets, all_preds)

    return running_loss / len(test_loader), test_nmse, all_preds, all_targets


# 训练循环
num_epochs = 100
best_nmse = float('inf')
best_model_path = 'best_model.pth'

train_losses = []
test_losses = []
train_nmses = []
test_nmses = []

for epoch in range(num_epochs):
    train_loss, train_nmse = train_model(model, train_loader, criterion, optimizer, device)
    test_loss, test_nmse, test_preds, test_targets = test_model(model, test_loader, criterion, device)

    # 更新学习率
    scheduler.step(test_loss)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_nmses.append(train_nmse)
    test_nmses.append(test_nmse)

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {train_loss:.6f}, Train NMSE: {train_nmse:.6f}, '
          f'Test Loss: {test_loss:.6f}, Test NMSE: {test_nmse:.6f}')

    # 保存最佳模型
    if test_nmse < best_nmse:
        best_nmse = test_nmse
        torch.save(model.state_dict(), best_model_path)
        print(f'保存最佳模型，测试NMSE: {best_nmse:.6f}')

print(f'训练完成，最佳测试NMSE: {best_nmse:.6f}')

# 加载最佳模型并进行最终测试
print("加载最佳模型进行最终测试...")
model.load_state_dict(torch.load(best_model_path))
final_test_loss, final_test_nmse, final_test_preds, final_test_targets = test_model(model, test_loader, criterion,
                                                                                    device)
print(f'最终测试NMSE: {final_test_nmse:.6f}')

# 可视化训练过程
plt.figure(figsize=(12, 10))

# 损失曲线
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()

# NMSE曲线
plt.subplot(2, 2, 2)
plt.plot(train_nmses, label='Train NMSE')
plt.plot(test_nmses, label='Test NMSE')
plt.xlabel('Epoch')
plt.ylabel('NMSE')
plt.title('Training and Test NMSE')
plt.legend()

# 预测结果可视化
plt.subplot(2, 1, 2)
plt.plot(final_test_targets[:100], label='True Values', alpha=0.7)
plt.plot(final_test_preds[:100], label='Predictions', alpha=0.7)
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Time Series Prediction Results')
plt.legend()

plt.tight_layout()
plt.savefig('training_and_results.png')
plt.show()

# 保存预测结果
np.save('test_targets.npy', final_test_targets)
np.save('test_preds.npy', final_test_preds)