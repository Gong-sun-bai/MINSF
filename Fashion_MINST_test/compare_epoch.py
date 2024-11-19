
#一、准备数据
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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

os.chdir('/home/gdut_students/lwb/Fashion_MINST_test')


# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

## 定义全连接网络
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(28*28,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,10)
    
    def forward(self,x):
        x = x.view(-1,28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNN()
model_MLP = MLP()


# 定义调用函数
def ori(model, name, logger, optimizer_type='SGD',batch_size = 700,learning_rate = 0.001,num_epochs = 100):
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # 超参数
    batch_size = batch_size
    learning_rate = learning_rate
    num_epochs = num_epochs

    # 加载 Fashion MNIST 数据集
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 根据输入选择优化器
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # 计算 total_steps
    total_steps = len(train_loader) * num_epochs

    # 初始化记录列表
    loss_history = []
    accuracy_history = []

    print(f"Running {name} model Experiment with {optimizer_type} optimizer...")
    logger.info(f"Running {name} model Experiment with {optimizer_type} optimizer...")

    # 开始训练循环
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累加损失
            total_loss += loss.item()

            # 打印训练日志
            if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # 计算并记录平均损失
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        # 每个 Epoch 结束后在测试集上评估模型性能
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            # 计算并记录测试集上的准确率
            accuracy = 100 * correct / total
            accuracy_history.append(accuracy)

            # 打印测试集上的准确率
            print(f'Accuracy of the model_{name} on the test images after Epoch [{epoch+1}/{num_epochs}]: {accuracy:.2f}%')
            logger.info(f'Accuracy of the model_{name} on the test images after Epoch [{epoch+1}/{num_epochs}]: {accuracy:.2f}%')

    # 绘制并保存 Loss 图像
    os.makedirs('./{images}', exist_ok=True)
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o', label='Loss')
    plt.title(f'{name} Loss Curve with {optimizer_type}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./images_net/{name}_loss_{optimizer_type}.png')
    plt.close()

    # 绘制并保存 Accuracy 图像
    plt.figure()
    plt.plot(range(1, num_epochs + 1), accuracy_history, marker='o', color='g', label='Accuracy')
    plt.title(f'{name} Accuracy Curve with {optimizer_type}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(f'./images_net/{name}_accuracy_{optimizer_type}.png')
    plt.close()

    # 打印最终的测试集准确率
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        final_accuracy = 100 * correct / total
        print(f'Final Accuracy of the model_{name} on the 10000 test images: {final_accuracy:.2f}% with {optimizer_type} optimizer')
        logger.info(f'Final Accuracy of the model_{name} on the 10000 test images: {final_accuracy:.2f}% with {optimizer_type} optimizer')

    print("Training complete.")
    logger.info("Training complete.")

if __name__ == "__main__":

    # 创建日志文件夹和图像保存文件夹
    images = 'images_net'
    logs = 'logs_net'
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists(images):
        os.makedirs(images)
    # 设置日志记录
    log_file = f'logs/{logs}.log'
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 记录每次实验的准确率
    cnn_accuracies = []
    mlp_accuracies = []

    # 运行 CNN 模型三次实验
    for i in range(3):
        print(f"\nRunning Experiment {i+1} for CNN model...")
        logger.info(f"\nRunning Experiment {i+1} for CNN model...")
        ori(model, 'CNN', logger, num_epochs=100)

        # 获取 CNN 模型的最终测试准确率
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in DataLoader(dataset=datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])), batch_size=700, shuffle=False):
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            final_accuracy = 100 * correct / total
            cnn_accuracies.append(final_accuracy)
            print(f'Final Accuracy of the CNN model on the test images: {final_accuracy:.2f}%')
            logger.info(f'Final Accuracy of the CNN model on the test images: {final_accuracy:.2f}%')

    # 运行 MLP 模型三次实验
    for i in range(3):
        print(f"\nRunning Experiment {i+1} for MLP model...")
        logger.info(f"\nRunning Experiment {i+1} for MLP model...")
        ori(model_MLP, 'MLP', logger, num_epochs=100)

        # 获取 MLP 模型的最终测试准确率
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in DataLoader(dataset=datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])), batch_size=700, shuffle=False):
                outputs = model_MLP(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            final_accuracy = 100 * correct / total
            mlp_accuracies.append(final_accuracy)
            print(f'Final Accuracy of the MLP model on the test images: {final_accuracy:.2f}%')
            logger.info(f'Final Accuracy of the MLP model on the test images: {final_accuracy:.2f}%')

    # 绘制对比图
    plt.figure()
    plt.plot([1, 2, 3], cnn_accuracies, marker='o', label='CNN Accuracy', color='b')
    plt.plot([1, 2, 3], mlp_accuracies, marker='o', label='MLP Accuracy', color='r')
    plt.title('Comparison of CNN and MLP Accuracy across 3 Experiments')
    plt.xlabel('Experiment Number')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid()
    plt.savefig(f'./images_net/accuracy_comparison.png')
    plt.show()

    # 打印最终的实验结果
    print("\nExperiment Results Summary:")
    for i in range(3):
        print(f"Experiment {i+1}: CNN Accuracy = {cnn_accuracies[i]:.2f}%, MLP Accuracy = {mlp_accuracies[i]:.2f}%")
        logger.info(f"Experiment {i+1}: CNN Accuracy = {cnn_accuracies[i]:.2f}%, MLP Accuracy = {mlp_accuracies[i]:.2f}%")

    print("All experiments completed.")
    logger.info("All experiments completed.")


