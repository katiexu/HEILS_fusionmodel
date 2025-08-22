from torchquantum.dataset import MNIST
import torch
import torch.nn.functional as F
import numpy as np
def transform(data):
    n=data.shape[0]
    m=data.shape[1]
    # Step 1: 分箱为 8 类 (0-7)
    bins = np.linspace(0, 1, 9)  # 8 个分箱边界 (0 到 1 的等间隔划分)
    data_binned = np.digitize(data, bins=bins) - 1  # 分箱结果 [0-7]

    # Step 2: 将每个整数转换为 3 维二进制向量
    # 创建预定义的二进制编码映射 (0-7 → 3-bit 向量)
    binary_map = {
        0: [0, 0, 0],
        1: [0, 0, 1],
        2: [0, 1, 0],
        3: [0, 1, 1],
        4: [1, 0, 0],
        5: [1, 0, 1],
        6: [1, 1, 0],
        7: [1, 1, 1]
    }

    # Step 3: 将分箱数据转换为 (n, 16, 3)
    # 初始化结果数组
    binary_data = np.zeros((n, 16, 3), dtype=np.uint8)

    # 遍历并编码每个值
    for i in range(n):
        for j in range(m):
            value = data_binned[i, j]
            if value==8:
                value=7
            binary_data[i, j] = binary_map[value]
    return binary_data
def MNISTDataLoaders(task,batch_size=32):
    digits = {'mnist01':[0,1],
              'mnist36':[3,6],
              'mnist4':[0,1,2,3],
              'mnist6':[0,1,2,3,4,5]
              }[task]

    dataset = MNIST(
        root='data',
        train_valid_split_ratio=[0.90, 0.1],
        center_crop=24,
        resize=28,
        resize_mode='bilinear',
        binarize=False,
        binarize_threshold=0.1307,
        digits_of_interest=digits,
        n_test_samples=None,
        n_valid_samples=None,
        fashion=False,
        n_train_samples=None
        )
    dataflow = dict()
    for split in dataset:
        if split == 'train':
            sampler = torch.utils.data.RandomSampler(dataset[split])
        else:
            # for valid and test, use SequentialSampler to make the train.py
            # and eval.py results consistent
            sampler = torch.utils.data.SequentialSampler(dataset[split])
            batch_size = len(dataset[split])

        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=True)

    return dataflow['train'], dataflow['valid'], dataflow['test']
def get_mnist_numpy(task,kernel=None):
    digits = {'mnist01': [0, 1],
              'mnist36': [3, 6],
              'mnist4': [0, 1, 2, 3],
              'mnist6': [0, 1, 2, 3, 4, 5],
              'mnist8': [0, 1, 2, 3, 4, 5,6,7],
              'mnist10': [0, 1, 2, 3, 4, 5,6,7,8,9],
              }[task]

    dataset = MNIST(
        root='data',
        train_valid_split_ratio=[0.90, 0.1],
        center_crop=24,
        resize=28,
        resize_mode='bilinear',
        binarize=False,
        binarize_threshold=0.1307,
        digits_of_interest=digits,
        n_test_samples=None,
        n_valid_samples=None,
        fashion=False,
        n_train_samples=None
    )

    # 提取 train/valid/test 数据并转换为 NumPy
    def extract_numpy(split):
        data_loader = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=len(dataset[split]),  # 一次性加载所有数据
            shuffle=True if split=='train' else False,
            pin_memory=True
        )
        # 获取一个 batch（全部数据）
        batch = next(iter(data_loader))
        images, labels = batch['image'], batch['digit']
        if kernel:
            images = F.avg_pool2d(images, kernel)
            images = images.view(images.shape[0], 24//kernel, 24//kernel).transpose(1, 2)
        # 转换为 NumPy
        images_np = images.numpy()  # (N, C, H, W)
        labels_np = labels.numpy()
        return images_np, labels_np

    # 返回 NumPy 数据
    train_images, train_labels = extract_numpy('train')
    valid_images, valid_labels = extract_numpy('valid')
    test_images, test_labels = extract_numpy('test')

    train_images=normalize_to_01(train_images)
    valid_images=normalize_to_01(valid_images)
    test_images=normalize_to_01(test_images)

    return (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels)


def normalize_to_01(arr):
    """将数组归一化到 [0,1] 区间"""
    min_val = np.min(arr)
    max_val = np.max(arr)

    # 避免除以零
    if max_val == min_val:
        return np.zeros_like(arr)

    normalized = (arr - min_val) / (max_val - min_val)
    return normalized
def one_hot(labels):
    # 获取所有唯一标签并创建索引映射
    unique_labels, indices = np.unique(labels, return_inverse=True)
    num_classes = len(unique_labels)

    # 使用映射后的索引进行one-hot编码
    one_hot = np.eye(num_classes)[indices]
    return one_hot
if __name__ == '__main__':
    train_datasets, val_datasets, test_datasets = get_mnist_numpy('mnist4')

    pass