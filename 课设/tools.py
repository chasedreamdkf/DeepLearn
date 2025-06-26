import os
import matplotlib.pyplot as plt


def plot_losses(items: dict):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 6))
    labels = items.keys()
    title = " and ".join(labels)
    for label, values in items.items():
        plt.plot(values, label=label + ' loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title} Loss Over Time')
    plt.legend()
    plt.grid(True)
    save_dir = './temp/imgs'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig('./temp/imgs/loss_plot.png')
    plt.show()