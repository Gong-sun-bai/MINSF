# 导入 AlexNet 模型
from torchvision.models import alexnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models  # 导入models
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import logging  # 导入logging模块

os.chdir('/home/gdut_students/lwb/MNIST_test')

# 创建日志文件夹和图像保存文件夹
if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists('images'):
    os.makedirs('images')

# 设置计算设备，选择GPU（cuda）如果可用，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理：包括将图像转换为Tensor，并进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
transform_alex = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 转为 3 通道
    transforms.Resize((224, 224)),  # 调整尺寸为 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载MNIST数据集
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
full_train_dataset_alex = datasets.MNIST(root='./data', train=True, download=True, transform=transform_alex)
test_dataset_alex = datasets.MNIST(root='./data', train=False, download=True, transform=transform_alex)


# 创建标签映射字典：将偶数和奇数标签映射到不同的类
even_label_map = {0: 0, 2: 1, 4: 2, 6: 3, 8: 4}
odd_label_map = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4}

# 筛选训练集：仅保留标签为偶数的图像，每类约6000张图像
train_indices = [i for i, (img, label) in enumerate(full_train_dataset) if label in [0, 2, 4, 6, 8]]
big_train_dataset = Subset(full_train_dataset, train_indices)
train_indices_alex = [i for i, (img, label) in enumerate(full_train_dataset_alex) if label in [0, 2, 4, 6, 8]]
big_train_dataset_alex = Subset(full_train_dataset_alex, train_indices_alex)

# 筛选small训练集：标签为奇数（1, 3, 5, 7, 9）的图像，每类仅取20张
small_train_indices = []
for label in [1, 3, 5, 7, 9]:
    indices = [i for i, (img, lbl) in enumerate(full_train_dataset) if lbl == label]
    small_train_indices.extend(indices[:20])  # 每类取20张图像
small_train_dataset = Subset(full_train_dataset, small_train_indices)
small_train_indices_alex = []
for label in [1, 3, 5, 7, 9]:
    indices = [i for i, (img, lbl) in enumerate(full_train_dataset_alex) if lbl == label]
    small_train_indices_alex.extend(indices[:20])  # 每类取20张图像
small_train_dataset_alex = Subset(full_train_dataset_alex, small_train_indices_alex)

# 筛选测试集：标签为奇数的图像，每类取200张
test_indices = []
for label in [1, 3, 5, 7, 9]:
    indices = [i for i, (img, lbl) in enumerate(test_dataset) if lbl == label]
    test_indices.extend(indices[:200])  # 每类取200张图像
test_dataset = Subset(test_dataset, test_indices)
test_indices_alex = []
for label in [1, 3, 5, 7, 9]:
    indices = [i for i, (img, lbl) in enumerate(test_dataset_alex) if lbl == label]
    test_indices_alex.extend(indices[:200])  # 每类取200张图像
test_dataset_alex = Subset(test_dataset_alex, test_indices_alex)


# 自定义数据集类：用于对标签进行映射
class RemapLabelsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, label_map):
        self.dataset = dataset
        self.label_map = label_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        new_label = self.label_map[label]
        return img, new_label

# 使用标签映射包装数据集
big_train_dataset = RemapLabelsDataset(big_train_dataset, even_label_map)
small_train_dataset = RemapLabelsDataset(small_train_dataset, odd_label_map)
test_dataset = RemapLabelsDataset(test_dataset, odd_label_map)
big_train_dataset_alex = RemapLabelsDataset(big_train_dataset_alex, even_label_map)
small_train_dataset_alex = RemapLabelsDataset(small_train_dataset_alex, odd_label_map)
test_dataset_alex = RemapLabelsDataset(test_dataset_alex, odd_label_map)

# 创建数据加载器
big_train_loader = DataLoader(big_train_dataset, batch_size=128, shuffle=True)
small_train_loader = DataLoader(small_train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
big_train_loader_alex = DataLoader(big_train_dataset_alex, batch_size=128, shuffle=True)
small_train_loader_alex = DataLoader(small_train_dataset_alex, batch_size=2, shuffle=True)
test_loader_alex = DataLoader(test_dataset_alex, batch_size=32, shuffle=False)


# 设置日志记录
log_file = f'logs_x/test1.log'
logger = logging.getLogger(log_file)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 定义卷积神经网络模型
class MNISTNet(nn.Module):
    def __init__(self, num_classes):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

# 定义使用 AlexNet 的迁移学习模型
class AlexNetTransfer(nn.Module):
    def __init__(self, num_classes):
        super(AlexNetTransfer, self).__init__()
        self.model = alexnet(weights='IMAGENET1K_V1')
        self.model.classifier[6] = nn.Linear(4096, num_classes)  # 修改输出层

    def forward(self, x):
        return self.model(x)

# 定义使用 LeakyReLU 激活函数的模型
class MNISTNetLeakyReLU(nn.Module):
    def __init__(self, num_classes):
        super(MNISTNetLeakyReLU, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, num_classes)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

# 改变数据集设置：增加小训练集的样本量（每类 100 张）
expanded_small_train_indices = []
for label in [1, 3, 5, 7, 9]:
    indices = [i for i, (img, lbl) in enumerate(full_train_dataset) if lbl == label]
    expanded_small_train_indices.extend(indices[:100])  # 每类取 100 张图像
expanded_small_train_dataset = Subset(full_train_dataset, expanded_small_train_indices)
expanded_small_train_dataset = RemapLabelsDataset(expanded_small_train_dataset, odd_label_map)
expanded_small_train_loader = DataLoader(expanded_small_train_dataset, batch_size=2, shuffle=True)

# 更新实验函数，添加新模型类型
def run_experiment(num_runs, model_type, big_train_loader, small_train_loader, test_loader, num_epochs,logger):
    all_accuracies = []
    all_losses = []
    
    for run in range(num_runs):
        logger.info(f"Run {run + 1}/{num_runs}")
        print(f"Run {run + 1}/{num_runs}")
        
        if model_type == 'alexnet_transfer':
            model = AlexNetTransfer(num_classes=5).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            train_model(model, big_train_loader_alex, criterion, optimizer, num_epochs, 'Transfer', logger)

            # 微调 AlexNet
            for param in model.model.features.parameters():
                param.requires_grad = False
            optimizer = optim.Adam(model.model.classifier[6].parameters(), lr=0.001)
            losses, accuracies = train_model(model, small_train_loader_alex, criterion, optimizer, num_epochs, 'Transfer', logger)

        elif model_type == 'leaky_relu':
            model = MNISTNetLeakyReLU(num_classes=5).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            losses, accuracies = train_model(model, small_train_loader, criterion, optimizer, num_epochs, 'LeakyReLU', logger)

        elif model_type == 'expanded_dataset':
            model = MNISTNet(num_classes=5).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            losses, accuracies = train_model(model, expanded_small_train_loader, criterion, optimizer, num_epochs, 'Expanded-Dataset', logger)
        
        elif model_type == 'non_transfer':
            model = MNISTNet(num_classes=5).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            print(f"Starting non-transfer training (run {run + 1})...")
            logger.info(f"Starting non-transfer training (run {run + 1})...")
            losses, accuracies = train_model(model, small_train_loader, criterion, optimizer, num_epochs, 'Non-Transfer',logger)
        
        elif model_type == 'transfer':
            # 训练大规模数据集（偶数类标签）上的模型
            model = MNISTNet(num_classes=5).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            logger.info(f"Starting transfer learning (run {run + 1})...")
            print(f"Starting transfer learning (run {run + 1})...")
            train_model(model, big_train_loader, criterion, optimizer, num_epochs, 'Transfer',logger)

            # 冻结卷积层的参数，进行微调
            for param in model.conv1.parameters():
                param.requires_grad = False
            for param in model.conv2.parameters():
                param.requires_grad = False

            model.fc1 = nn.Linear(64 * 7 * 7, 5).to(device)

            # 只更新需要训练的参数
            params_to_train = [param for param in model.parameters() if param.requires_grad]
            optimizer = optim.Adam(params_to_train, lr=0.001)

            # 微调
            losses, accuracies = train_model(model, small_train_loader, criterion, optimizer, num_epochs, 'Transfer',logger)
      
        # 测试
        accuracy = test_model(model, test_loader)
        logger.info(f'{model_type.capitalize()} Test Accuracy: {accuracy:.2f}%')
        all_accuracies.append(accuracy)
        all_losses.append(losses)

        # 保存结果图
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(accuracies, label=f'Run {run + 1}')
        plt.title(f'{model_type.capitalize()} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(losses, label=f'Run {run + 1}')
        plt.title(f'{model_type.capitalize()} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.savefig(f'images/{model_type}_{run + 1}_performance.png')
        plt.close()

    return all_accuracies, all_losses

# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs, experiment_type,logger):
    train_losses = []
    train_accuracies = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        model.train()  # 设置模型为训练模式
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_losses.append(epoch_loss / len(train_loader))
        train_accuracies.append(100 * correct / total)
         # 记录日志
        if logger:
            logger.info(f'[{experiment_type}] Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')

        print(f'[{experiment_type}] Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')
    
    return train_losses, train_accuracies

# 测试函数
def test_model(model, test_loader):
    correct = 0
    total = 0
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# 可视化所有实验结果
def plot_results(accuracies, losses, title):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(accuracies)
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Run')
    plt.ylabel('Accuracy (%)')
    
    plt.subplot(1, 2, 2)
    plt.plot(np.mean(losses, axis=1))
    plt.title(f'{title} - Loss')
    plt.xlabel('Run')
    plt.ylabel('Loss')
    
    plt.savefig(f'images_x/{title}_final_performance.png')
    plt.close()

# 运行新实验
num_runs = 1
num_epochs = 2

# 非迁移学习实验
print("Running Non-Transfer Experiment...")
logger.info("Running Non-Transfer Experiment...")
non_transfer_accuracies, non_transfer_losses = run_experiment(num_runs, 'non_transfer', big_train_loader, small_train_loader, test_loader, num_epochs,logger)

# 迁移学习实验
print("Running Transfer Experiment...")
logger.info("Running Transfer Experiment...")
transfer_accuracies, transfer_losses = run_experiment(num_runs, 'transfer', big_train_loader, small_train_loader, test_loader, num_epochs,logger)

print("Running Expanded Dataset Experiment...")
logger.info("Running Expanded Dataset Experiment...")
expanded_dataset_accuracies, expanded_dataset_losses = run_experiment(num_runs, 'expanded_dataset', big_train_loader, expanded_small_train_loader, test_loader, num_epochs,logger)

print("Running LeakyReLU Experiment...")
logger.info("Running LeakyReLU Experiment...")
leaky_relu_accuracies, leaky_relu_losses = run_experiment(num_runs, 'leaky_relu', big_train_loader, small_train_loader, test_loader, num_epochs,logger)

print("Running AlexNet Transfer Experiment...")
logger.info("Running AlexNet Transfer Experiment...")
alexnet_accuracies, alexnet_losses = run_experiment(num_runs, 'alexnet_transfer', big_train_loader_alex, small_train_loader_alex, test_loader_alex, num_epochs,logger)




# 保存新的结果
np.save('logs/alexnet_accuracies.npy', alexnet_accuracies)
np.save('logs/alexnet_losses.npy', alexnet_losses)
np.save('logs/leaky_relu_accuracies.npy', leaky_relu_accuracies)
np.save('logs/leaky_relu_losses.npy', leaky_relu_losses)
np.save('logs/expanded_dataset_accuracies.npy', expanded_dataset_accuracies)
np.save('logs/expanded_dataset_losses.npy', expanded_dataset_losses)
# 保存结果
np.save('logs/non_transfer_accuracies.npy', non_transfer_accuracies)
np.save('logs/non_transfer_losses.npy', non_transfer_losses)
np.save('logs/transfer_accuracies.npy', transfer_accuracies)
np.save('logs/transfer_losses.npy', transfer_losses)


# 绘制所有实验结果
plot_results(alexnet_accuracies, alexnet_losses, 'AlexNet Transfer Experiment')
plot_results(leaky_relu_accuracies, leaky_relu_losses, 'LeakyReLU Experiment')
plot_results(expanded_dataset_accuracies, expanded_dataset_losses, 'Expanded Dataset Experiment')
# 绘制迁移和非迁移实验的结果
plot_results(non_transfer_accuracies, non_transfer_losses, 'Non-Transfer Experiment')
plot_results(transfer_accuracies, transfer_losses, 'Transfer Experiment')


print("All experiments completed and results saved.")
logger.info("All experiments completed and results saved.")