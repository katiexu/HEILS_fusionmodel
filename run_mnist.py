import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from FusionModel import translator,TQLayer
import os


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


# 加载和预处理数据
def load_data(dataset_name):
    from dataset import get_mnist_numpy, transform

    train_datasets, val_datasets, test_datasets = get_mnist_numpy(dataset_name, 6)
    X_train, y_train = train_datasets
    X_test, y_test = test_datasets

    # 重塑数据
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    # 转换数据
    X_train = transform(X_train)
    X_test = transform(X_test)

    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    return (X_train_tensor, y_train_tensor), (X_test_tensor, y_test_tensor)


# 加载数据
(X_train, y_train), (X_test, y_test) = load_data('mnist01')

# 创建数据加载器
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型
n_qubits = 6
n_layers = 4
seq_length = 16
n_class = 2

single = [[i] + [1] * 2 * n_layers for i in range(1, n_qubits + 1)]
enta = [[i] + [i + 1] * n_layers for i in range(1, n_qubits)] + [[n_qubits] + [1] * n_layers]

design = translator(single, enta, 'full', (n_qubits, n_layers), 1)
model = TQLayer(n_qubits, design, seq_length, n_class).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)


# 训练函数
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return running_loss / len(train_loader), accuracy


# 测试函数
def test_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return running_loss / len(test_loader), accuracy


# 训练循环
num_epochs = 50
best_acc = 0.0
best_model_path = 'best_model.pth'

train_losses = []
test_losses = []
train_accs = []
test_accs = []

for epoch in range(num_epochs):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test_model(model, test_loader, criterion, device)

    # 更新学习率
    scheduler.step(test_acc)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), best_model_path)
        print(f'保存最佳模型，准确率: {best_acc:.4f}')

print(f'训练完成，最佳测试准确率: {best_acc:.4f}')

# 加载最佳模型并进行最终测试
print("加载最佳模型进行最终测试...")
model.load_state_dict(torch.load(best_model_path))
final_test_loss, final_test_acc = test_model(model, test_loader, criterion, device)
print(f'最终测试准确率: {final_test_acc:.4f}')

# 可视化训练过程
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_history.png')
plt.show()