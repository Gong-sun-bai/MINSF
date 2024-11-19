import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

# 设置计算设备，选择GPU（cuda）如果可用，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理：包括将图像转换为Tensor，并进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载MNIST数据集
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建标签映射字典：将偶数和奇数标签映射到不同的类
even_label_map = {0: 0, 2: 1, 4: 2, 6: 3, 8: 4}
odd_label_map = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4}

# 筛选训练集：仅保留标签为偶数的图像，每类约6000张图像
train_indices = [i for i, (img, label) in enumerate(full_train_dataset) if label in [0, 2, 4, 6, 8]]
big_train_dataset = Subset(full_train_dataset, train_indices)

# 筛选small训练集：标签为奇数（1, 3, 5, 7, 9）的图像，每类仅取20张
small_train_indices = []
for label in [1, 3, 5, 7, 9]:
    indices = [i for i, (img, lbl) in enumerate(full_train_dataset) if lbl == label]
    small_train_indices.extend(indices[:20])  # 每类取20张图像
small_train_dataset = Subset(full_train_dataset, small_train_indices)

# 筛选测试集：标签为奇数的图像，每类取200张
test_indices = []
for label in [1, 3, 5, 7, 9]:
    indices = [i for i, (img, lbl) in enumerate(test_dataset) if lbl == label]
    test_indices.extend(indices[:200])  # 每类取200张图像
test_dataset = Subset(test_dataset, test_indices)

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

# 创建数据加载器
big_train_loader = DataLoader(big_train_dataset, batch_size=128, shuffle=True)
small_train_loader = DataLoader(small_train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs):
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
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')
    
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

# 多次实验
def run_experiment(num_runs, model_type, big_train_loader, small_train_loader, test_loader, num_epochs):
    all_accuracies = []
    all_losses = []
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        model = MNISTNet(num_classes=5).to(device)
        
        if model_type == 'non_transfer':
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            losses, accuracies = train_model(model, small_train_loader, criterion, optimizer, num_epochs)
        
        elif model_type == 'transfer':
            # 训练大规模数据集（偶数类标签）上的模型
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            train_model(model, big_train_loader, criterion, optimizer, num_epochs)
            
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
            losses, accuracies = train_model(model, small_train_loader, criterion, optimizer, num_epochs)
        
        # 测试
        accuracy = test_model(model, test_loader)
        print(f'Test Accuracy: {accuracy:.2f}%')
        all_accuracies.append(accuracy)
        all_losses.append(losses)
    
    return all_accuracies, all_losses

# 进行10次实验
num_runs = 10
num_epochs = 10
non_transfer_accuracies, non_transfer_losses = run_experiment(num_runs, 'non_transfer', big_train_loader, small_train_loader, test_loader, num_epochs)
transfer_accuracies, transfer_losses = run_experiment(num_runs, 'transfer', big_train_loader, small_train_loader, test_loader, num_epochs)

# 可视化结果
def plot_results(accuracies, losses, title):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(accuracies)
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Run')
    plt.ylabel('Accuracy (%)')
    
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title(f'{title} - Loss')
    plt.xlabel('Run')
    plt.ylabel('Loss')
    
    plt.show()

# 绘制迁移和非迁移实验的结果
plot_results(non_transfer_accuracies, np.mean(non_transfer_losses, axis=1), 'Non-Transfer Experiment')
plot_results(transfer_accuracies, np.mean(transfer_losses, axis=1), 'Transfer Experiment')
