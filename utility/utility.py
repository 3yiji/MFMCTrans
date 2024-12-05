import h5py
import numpy as np
import os
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import fft, fftfreq, fftshift
from config import project_path
from matplotlib.ticker import MultipleLocator
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm


def confusionMatrix(y_true, y_pred, classes, figsize=(8, 6)):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    # 计算每个类别的正确率
    # cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 创建热力图
    fig = plt.figure(figsize=figsize)
    ax = sns.heatmap(cm, annot=False, cmap='Blues', fmt='.1', xticklabels=classes, yticklabels=classes, linewidths=0.5, linecolor='black')
    fig.add_axes(ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    # ax.set_title('Confusion Matrix')
    fig.tight_layout()

    fig.show()
    return fig


def fig_accuracy(model, dataLoader, dBs, criterion, batch_size=128, isshow=True, dBconfustion=-100, aug=1):
    fig, ax = plt.subplots()
    ax.set_xlabel('Signal to noise ratio [dB]')
    ax.set_ylabel('Correct classification probability')
    model.eval()
    accuracies = []
    total_loss = 0
    n_all = 0
    for dB in dBs:
        data_test, label_test = dataLoader.load_dB(dB)
        torch.cuda.empty_cache()
        outputs = model_batch(model, data_test, batch_size=batch_size)

        # 为训练时的数据增强所作的操作test-time-augmentation
        outputs = torch.mean(outputs.reshape(-1, aug, outputs.shape[1]), dim=1)
        # outputs, _ = torch.max(outputs.reshape(-1, aug, outputs.shape[1]), dim=1)
        label_test = label_test[::aug]

        total_loss += criterion(outputs, label_test).item()*outputs.shape[0]
        n_all += outputs.shape[0]
        y_true = label_test.to(device='cpu').numpy()
        y_pred = np.argmax(outputs.to(device='cpu').numpy(), axis=1)
        if dB == dBconfustion:
            utility.confusionMatrix(y_true, y_pred, dataLoader.classes)
        accuracy = accuracy_score(y_true, y_pred)
        accuracies.append(accuracy)
    total_loss /= n_all

    # 画图
    ax.plot(dBs, accuracies, marker='*')
    ax.yaxis.set_major_locator(MultipleLocator(0.1))  # 主要刻度间隔为 1
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))  # 次要刻度间隔为 0.1
    ax.grid(True)
    if isshow:
        fig.show()
    else:
        plt.close(fig)
    # return fig
    return fig, np.stack([dBs, accuracies], axis=0), total_loss


def model_batch(model, x, batch_size=128):
    y = []
    for start_idx in range(0, x.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, x.shape[0])
        y.append(model(x[start_idx:end_idx]))
    y = torch.concatenate(y, dim=0)
    return y


if __name__ == '__main__':
    pass
