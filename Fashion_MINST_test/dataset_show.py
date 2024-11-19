import matplotlib.pyplot as plt
from torchvision import datasets, transforms
dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
classes = dataset.classes
plt.figure(figsize=(12, 6))
for i in range(10):
    image, label = dataset[i]
    plt.subplot(2, 5, i + 1)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(classes[label])
    plt.axis("off")
plt.tight_layout()
plt.show()  # 显示图片
plt.savefig(f'images/final_performance.png')
plt.close()