import numpy as np
import matplotlib.pyplot as plt

# 加载已保存的实验数据
non_transfer_accuracies = np.load('logs/non_transfer_accuracies.npy')
non_transfer_losses = np.load('logs/non_transfer_losses.npy')

transfer_accuracies = np.load('logs/transfer_accuracies.npy')
transfer_losses = np.load('logs/transfer_losses.npy')

alexnet_accuracies = np.load('logs/alexnet_accuracies.npy')
alexnet_losses = np.load('logs/alexnet_losses.npy')

leaky_relu_accuracies = np.load('logs/leaky_relu_accuracies.npy')
leaky_relu_losses = np.load('logs/leaky_relu_losses.npy')

expanded_dataset_accuracies = np.load('logs/expanded_dataset_accuracies.npy')
expanded_dataset_losses = np.load('logs/expanded_dataset_losses.npy')

# 绘制所有实验结果
def plot_results(accuracies, losses, title):
    plt.figure(figsize=(12, 5))
    
    # 绘制 Accuracy 图
    plt.subplot(1, 2, 1)
    plt.plot(accuracies, marker='o', color='b', label='Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Run')
    plt.ylabel('Accuracy (%)')
    
    # 在每个点上标注 Accuracy 数值
    for i, acc in enumerate(accuracies):
        plt.annotate(f'{acc:.2f}', (i, acc), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=10)

    plt.legend()

    # 绘制 Loss 图
    plt.subplot(1, 2, 2)
    mean_losses = np.mean(losses, axis=1)
    plt.plot(mean_losses, marker='o', color='r', label='Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Run')
    plt.ylabel('Loss')
    
    # 在每个点上标注 Loss 数值
    for i, loss in enumerate(mean_losses):
        plt.annotate(f'{loss:.4f}', (i, loss), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=10)

    plt.legend()

    # 保存图像并关闭
    plt.savefig(f'images/{title}_final_performance.png')
    plt.close()

# 绘制实验结果
plot_results(non_transfer_accuracies, non_transfer_losses, 'Non-Transfer Experiment')
plot_results(transfer_accuracies, transfer_losses, 'Transfer Experiment')
plot_results(alexnet_accuracies, alexnet_losses, 'AlexNet Transfer Experiment')
plot_results(leaky_relu_accuracies, leaky_relu_losses, 'LeakyReLU Experiment')
plot_results(expanded_dataset_accuracies, expanded_dataset_losses, 'Expanded Dataset Experiment')
