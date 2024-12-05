from matplotlib import pyplot as plt
from utility.utility import fig_accuracy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import Proposed
from config import project_path, sig53_impaired_path, sig53_impaired_aug_path, sig53_clean_aug_path
import sys
from sklearn.metrics import accuracy_score
from scipy.signal import windows
from tqdm import tqdm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as patches
import Loader


# 评价效果
def main():
    np.set_printoptions(suppress=True)
    batch_size = 64
    path_dataset = sig53_impaired_aug_path
    path_index_test = project_path + 'index/sig53_impaired_aug/index_test.npy'

    file_name = '57_2024-11-01_16-50_0.565_0.577_0.835_0.750'
    device = 'cuda'
    # 加载数据迭代器
    dataLoader_test = Loader.DataLoader(path_dataset, path_index_test, batch_size=batch_size, device=device,
                                        isPreLoad=True)
    # 创建模型实例
    model = Proposed.Proposed().to(device=device)
    dict = torch.load(project_path + 'weights/Proposed/Sig53/impaired/翻转互换128步2头6层多模态/' + file_name + '.pth')
    model.load_state_dict(dict)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 查看测试的偏差
    model.eval()                # 评估模式
    with torch.no_grad():
        torch.cuda.empty_cache()

        _, accuracys, e_loss = fig_accuracy(model, dataLoader_test, list(range(-1, 31, 1)), criterion, batch_size=batch_size*1, isshow=True, aug=4)
        np.save(project_path+'results/data/'+file_name+'.npy', accuracys)
        accuracy = accuracys[1][np.where(accuracys[0] == 30)][0]
        print(f'Eloss {e_loss:.3f}, Accuracy {accuracy:.4f}, Accuracys_mean {np.mean(accuracys[1]):.4f}')
        print(np.array2string(accuracys, precision=4, max_line_width=400))

        fig, ax = plt.subplots()
        data = np.mean(accuracys.reshape(2, -1, 1), axis=2)
        ax.plot(data[0], data[1])


# 评价效果
def main_clean():
    np.set_printoptions(suppress=True)
    batch_size = 64
    path_dataset = sig53_clean_aug_path
    path_index_test = project_path + 'index/sig53_clean_aug/index_test.npy'

    file_name = '28_2024-12-03_18-22_0.003_0.000_1.000_1.000'
    device = 'cuda'
    # 加载数据迭代器
    dataLoader_test = Loader.DataLoader(path_dataset, path_index_test, batch_size=batch_size, device=device,
                                        isPreLoad=True)
    # 创建模型实例
    model = Proposed.Proposed().to(device=device)
    dict = torch.load(project_path + 'weights/Proposed/Sig53/clean/翻转互换128步2头6层多模态/' + file_name + '.pth')
    model.load_state_dict(dict)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 查看测试的偏差
    model.eval()                # 评估模式
    with torch.no_grad():
        torch.cuda.empty_cache()

        _, accuracys, e_loss = fig_accuracy(model, dataLoader_test, list(range(100, 101, 1)), criterion, batch_size=batch_size*1, isshow=True, aug=4)
        # np.save(project_path+'results/data/'+file_name+'.npy', accuracys)
        accuracy = accuracys[1][np.where(accuracys[0] == 100)][0]
        print(f'Eloss {e_loss:.10f}, Accuracy {accuracy:.10f}, Accuracys_mean {np.mean(accuracys[1]):.10f}')
        print(np.array2string(accuracys, precision=4, max_line_width=400))

        fig, ax = plt.subplots()
        data = np.mean(accuracys.reshape(2, -1, 1), axis=2)
        ax.plot(data[0], data[1])

# 画出数据增强的示意图
def augmentation_fig():
    path_dataset = sig53_impaired_aug_path
    path_index_test = project_path + 'index/sig53_impaired_aug/index_test.npy'
    dataLoader_test = Loader.DataLoader(path_dataset, path_index_test)
    data, _ = dataLoader_test.load_snr_cls_one(30, 16)
    data = data[1000:1032]

    cmap_i = plt.get_cmap('Blues')
    cmap_q = plt.get_cmap('Reds')
    norm = Normalize(vmin=-data.shape[0]*0.5, vmax=data.shape[0]-1)
    data_norm = norm(np.arange(data.shape[0])).reshape(-1, 1)
    data = np.concatenate([data, data_norm], axis=1)

    # 画出图形
    orders = ['(a)', '(b)', '(c)', '(d)']
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for j in range(4):
        ax = axs[j//2, j%2]
        dat = data[::-1] if j//2 else data
        steps = np.arange(dat.shape[0])
        for i in range(dat.shape[0]):
            ax.plot(steps[i:i+2], dat[i:i+2, j%2], color=cmap_i(dat[i, 2]), linewidth=None)
            ax.plot(steps[i:i+2], dat[i:i+2, (j+1)%2], color=cmap_q(dat[i, 2]), linewidth=None)
        ax.set_facecolor('black')
        ax.text(0.5, -0.05, orders[j], transform=ax.transAxes, fontsize=30, fontweight='normal', va='top', ha='left')
        ax.set_xticks([])  # 关闭 x 轴刻度
        ax.set_yticks([])  # 关闭 y 轴刻度
    fig.tight_layout()
    fig.savefig(project_path + 'results/figures/augmentation.svg')
    fig.show()

# 画出stft变换的图
def stft_fig():
    path_dataset = sig53_impaired_aug_path
    path_index_test = project_path + 'index/sig53_impaired_aug/index_test.npy'
    dataLoader_test = Loader.DataLoader(path_dataset, path_index_test)
    data0, _ = dataLoader_test.load_snr_cls_one(30, 0)
    data1, _ = dataLoader_test.load_snr_cls_one(30, 29)
    data = np.stack([data0, data1], axis=0)
    data = torch.tensor(data)
    title = ['4ASK', '4FSK']

    x_commplex = data[:, :, 0] + data[:, :, 1] * 1j
    window = torch.hann_window(32)  # 使用 Hann 窗函数
    stft_result = torch.stft(x_commplex, n_fft=64, hop_length=32, win_length=32,
                             window=window,
                             return_complex=True)[:, :, :128]
    stft_result = torch.abs(stft_result)
    stft_result = torch.fft.fftshift(stft_result, dim=1)
    stft_result = stft_result.permute(0, 2, 1)

    x = np.arange(stft_result.shape[2])
    y = np.arange(stft_result.shape[1])
    X, Y = np.meshgrid(x, y)

    orders = ['(a)', '(b)']
    fig = plt.figure(figsize=(6, 12))
    # axes = fig.add_subplot(1, 2, projection='3d')
    for i, result in enumerate(stft_result):
        ax = fig.add_subplot(2, 1, i+1, projection='3d')

        ax.plot_surface(X, Y, result, cmap='viridis')

        ax.set_xlabel('Frequency')
        ax.set_ylabel(' Time ')
        ax.set_zlabel('Magnitude')

        ax.view_init(30, -45)
        ax.set_box_aspect([1, 2, 1])
        ax.text2D(0.5, 0, s=orders[i], transform=ax.transAxes, fontsize=20, fontweight='normal', va='center',
                ha='center')

    fig.tight_layout()
    fig.subplots_adjust(right=0.92)
    fig.savefig(project_path + 'results/figures/stft.svg')
    fig.show()


# 画出IQ与AP的示意图
def IQAP_fig():
    path_dataset = sig53_impaired_aug_path
    path_index_test = project_path + 'index/sig53_impaired_aug/index_test.npy'
    dataLoader_test = Loader.DataLoader(path_dataset, path_index_test)
    data, _ = dataLoader_test.load_snr_cls_one(30, 16)
    data = data[1000:1032]
    pos1 = 18
    pos2 = 23

    # 生成IQ信号
    I = data[:, 0]
    Q = data[:, 1]

    # 随机选择两个点进行标记
    I1, Q1 = I[pos1], Q[pos1]
    I2, Q2 = I[pos2], Q[pos2]

    # 计算幅度和相位
    A1, P1 = np.sqrt(I1 ** 2 + Q1 ** 2), np.arctan2(Q1, I1)
    A2, P2 = np.sqrt(I2 ** 2 + Q2 ** 2), np.arctan2(Q2, I2)

    # 绘制星座图
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(I, Q, label="Constellation Points", alpha=0.7)
    ax.scatter([I1, I2], [Q1, Q2], color='red', label="Selected Points")

    # 标记选中的点（使用(I, Q)和(A, P)表示）
    ax.text(I1, Q1+0.08, f"(I={I1:.2f}, Q={Q1:.2f})", color='blue', ha='left', fontsize=11)
    ax.text(I2, Q2, f"(I={I2:.2f}, Q={Q2:.2f})", color='blue', ha='left', va='top', fontsize=11)
    ax.text(I1, Q1, f"(A={A1:.2f}, P={P1:.2f})", color='purple', ha='left', fontsize=11)
    ax.text(I2, Q2-0.08, f"(A={A2:.2f}, P={P2:.2f})", color='purple', ha='left', va='top', fontsize=11)

    # 添加 I 和 Q 的虚线
    ax.set_xticks(np.arange(-0.9, 1, 0.2))
    ax.set_yticks(np.arange(-0.9, 1, 0.2))
    # 获取图形的当前边界
    x_left, x_right = ax.get_xlim()
    y_bottom, y_top = ax.get_ylim()
    # 手动设置边界以确保绘制到边缘
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)
    ax.hlines(y=Q1, xmin=min(I1, x_left), xmax=max(I1, x_left), color='orange', linestyle='--', linewidth=0.5, label="I/Q Axis")
    ax.vlines(x=I1, ymin=min(Q1, y_bottom), ymax=max(Q1, y_bottom), color='orange', linestyle='--', linewidth=0.5)
    ax.hlines(y=Q2, xmin=min(I2, x_left), xmax=max(I2, x_left), color='orange', linestyle='--', linewidth=0.5)
    ax.vlines(x=I2, ymin=min(Q2, y_bottom), ymax=max(Q2, y_bottom), color='orange', linestyle='--', linewidth=0.5)

    # 添加幅度的圆弧
    arc1 = patches.Arc((0, 0), 2 * A1, 2 * A1, theta1=0, theta2=np.degrees(P1), color="green", linestyle='-.',
                       linewidth=1, label="Amplitude (A)")
    arc2 = patches.Arc((0, 0), 2 * A2, 2 * A2, theta1=0, theta2=np.degrees(P2), color="green", linestyle='-.',
                       linewidth=1)
    ax.add_patch(arc1)
    ax.add_patch(arc2)

    # 添加相位的放射线
    ax.plot([0, A1*np.cos(P1)], [0, A1*np.sin(P1)], color='black', linestyle=':', linewidth=0.8)
    ax.plot([0, A2*np.cos(P2)], [0, A2*np.sin(P2)], color='black', linestyle=':', linewidth=0.8)

    # 设置图表属性
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel("I (In-phase)")
    ax.set_ylabel("Q (Quadrature)")
    # plt.grid()
    fig.tight_layout()
    fig.savefig(project_path + 'results/figures/IQAP.svg')
    fig.show()


# 画出一个信号的图
def fig_sig53_one():
    pos = np.arange(0, 128)
    cls = 34
    path_dataset = sig53_impaired_aug_path
    path_index_test = project_path + 'index/sig53_impaired_aug/index_test.npy'
    dataLoader_test = Loader.DataLoader(path_dataset, path_index_test)
    sigdata = dataLoader_test.load_snr_cls_one(30, cls)[0][pos]
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(sigdata, linewidth=6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xticks([])  # 关闭 x 轴刻度
    ax.set_yticks([])  # 关闭 y 轴刻度

    fig.tight_layout()
    fig.show()
    fig.savefig(project_path + 'results/figures/fig_sig53.svg')


# 比较是否使用TTA的效果
def test_time_augmentation_compare():
    np.set_printoptions(suppress=True)
    batch_size = 64

    file_name = '57_2024-11-01_16-50_0.565_0.577_0.835_0.750'
    device = 'cuda'
    # 加载数据迭代器
    path_dataset = sig53_impaired_path
    path_index_test = project_path + 'index/sig53_impaired/index_test.npy'
    dataLoader_test = Loader.DataLoader(path_dataset, path_index_test, batch_size=batch_size, device=device, isPreLoad=True)
    # 创建模型实例
    dict = torch.load(project_path + 'weights/Proposed/Sig53/impaired/翻转互换128步2头6层多模态/' + file_name + '.pth')
    model = Proposed.Proposed(device=device).to(device=device)
    model.load_state_dict(dict)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 查看测试的偏差
    model.eval()  # 评估模式
    with torch.no_grad():
        torch.cuda.empty_cache()

        _, accuracys, e_loss = fig_accuracy(model, dataLoader_test, list(range(-1, 31, 1)), criterion,
                                            batch_size=batch_size * 1, isshow=True, aug=1)
        np.save(project_path + 'results/data/' + 'noTTA_'+ file_name + '.npy', accuracys)
        accuracy = accuracys[1][np.where(accuracys[0] == 30)][0]
        print(f'Eloss {e_loss:.3f}, Accuracy {accuracy:.4f}, Accuracys_mean {np.mean(accuracys[1]):.4f}')
        print(np.array2string(accuracys, precision=4, max_line_width=400))

        fig, ax = plt.subplots()
        data = np.mean(accuracys.reshape(2, -1, 1), axis=2)
        ax.plot(data[0], data[1])


# 计算函数的参数量
def cal_params():
    from models import Proposed
    model = MyTrans8.MyTrans8()
    total_params = sum(p.numel() for p in model.parameters())
    total_params = total_params / 1e6
    print('total_params = %.1f' % total_params)


# 评估内部家族的精度
def cal_family_accuracy():
    from utility.utility import model_batch

    np.set_printoptions(suppress=True)
    batch_size = 64

    clas = [
        [3, 6, 10, 15, 19],  # ASK
        [0, 2, 5, 9, 14, 18],  # PAM
        [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],  # FSK
        [1, 4, 7, 11, 16, 20],  # PSK
        [8, 12, 13, 17, 21, 22, 23, 24],  # QAM
        [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]  # OFDM
    ]
    # file_name = '58_2024-11-01_21-06_0.564_0.577_0.839_0.749'
    file_name = '57_2024-11-01_16-50_0.565_0.577_0.835_0.750'
    device = 'cuda'
    # 加载数据迭代器
    path_dataset = sig53_impaired_aug_path
    path_index_test = project_path + 'index/sig53_impaired_aug/index_test.npy'
    dataLoader_test = Loader.DataLoader(path_dataset, path_index_test, batch_size=batch_size, device=device, isPreLoad=True)
    # 创建模型实例
    model = Proposed.Proposed().to(device=device)
    dict = torch.load(project_path + 'weights/Proposed/Sig53/impaired/翻转互换128步2头6层多模态/' + file_name + '.pth')
    model.load_state_dict(dict)
    accuracies = []
    # 查看测试的偏差
    model.eval()  # 评估模式
    with torch.no_grad():
        torch.cuda.empty_cache()

        for cla in clas:
            datax, datay = dataLoader_test.load_mul_cla(cla)

            outputs = model_batch(model, datax, batch_size=batch_size)

            # 为训练时的数据增强所作的操作test-time-augmentation
            outputs = torch.mean(outputs.reshape(-1, 4, outputs.shape[1]), dim=1)
            label_test = datay[::4]

            y_true = label_test.to(device='cpu').numpy()
            y_pred = np.argmax(outputs.to(device='cpu').numpy(), axis=1)
            accuracy = accuracy_score(y_true, y_pred)
            accuracies.append(accuracy)
        print(np.array2string(np.array(accuracies), precision=4, max_line_width=400))


# 评估家族之间的精度，且画出混淆矩阵
def cal_interfamily_accuracy():
    from utility.utility import model_batch
    import os
    from sklearn.metrics import confusion_matrix, accuracy_score
    import seaborn as sns


    path_y_pred = project_path + 'tem/y_pred.npy'
    path_y_true = project_path + 'tem/y_true.npy'

    clas = [
        [3, 6, 10, 15, 19],  # ASK
        [0, 2, 5, 9, 14, 18],  # PAM
        [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],  # FSK
        [1, 4, 7, 11, 16, 20],  # PSK
        [8, 12, 13, 17, 21, 22, 23, 24],  # QAM
        [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]  # OFDM
    ]

    if not os.path.exists(path_y_pred):
        np.set_printoptions(suppress=True)
        batch_size = 64

        # file_name = '58_2024-11-01_21-06_0.564_0.577_0.839_0.749'
        file_name = '57_2024-11-01_16-50_0.565_0.577_0.835_0.750'
        device = 'cuda'
        # 加载数据迭代器
        path_dataset = sig53_impaired_aug_path
        path_index_test = project_path + 'index/sig53_impaired_aug/index_test.npy'
        dataLoader_test = Loader.DataLoader(path_dataset, path_index_test, batch_size=batch_size, device=device,
                                            isPreLoad=True)
        # 创建模型实例
        dict = torch.load(project_path + 'weights/Proposed/Sig53/impaired/翻转互换128步2头6层多模态/' + file_name + '.pth')
        model = Proposed.Proposed().to(device=device)
        model.load_state_dict(dict)

        # 查看测试的偏差
        model.eval()  # 评估模式
        with torch.no_grad():
            torch.cuda.empty_cache()
            test_datay = []
            outputs_datay = []
            for i, cla in enumerate(clas):
                datax, datay = dataLoader_test.load_mul_cla(cla)
                test_datay.append(datay)

                outputs = model_batch(model, datax, batch_size=batch_size)
                outputs_datay.append(outputs)
            test_datay = torch.concatenate(test_datay, dim=0)
            outputs_datay = torch.concatenate(outputs_datay, dim=0)

            # 为训练时的数据增强所作的操作test-time-augmentation
            outputs = torch.mean(outputs_datay.reshape(-1, 4, outputs.shape[1]), dim=1)
            label_test = test_datay[::4]

            y_true = label_test.to(device='cpu').numpy()
            y_pred = np.argmax(outputs.to(device='cpu').numpy(), axis=1)
            np.save(path_y_true, y_true)
            np.save(path_y_pred, y_pred)

    else:
        y_true = np.load(path_y_true)
        y_pred = np.load(path_y_pred)

    from index.sig53_impaired_aug import classes
    classes = classes.classes
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    # 计算每个类别的正确率
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # 创建热力图
    fig = plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_percentage, annot=False, cmap='Blues', fmt='.1', xticklabels=classes, yticklabels=classes,
                     linewidths=0.5, linecolor='black')
    fig.add_axes(ax)
    ax.set_xlabel('Predicted label', fontsize=15)
    ax.set_ylabel('True label', fontsize=15)
    fig.tight_layout()
    fig.show()
    fig.savefig(project_path + 'results/figures/cal_all_accuracy.svg')


    y_true_trans = np.empty_like(y_true)
    y_pred_trans = np.empty_like(y_pred)
    for i, cla in enumerate(clas):
        y_true_trans[np.isin(y_true, cla)] = i
        y_pred_trans[np.isin(y_pred, cla)] = i

    classes = ['ask', 'pam', 'fsk', 'psk', 'qam', 'ofdm']
    # 计算混淆矩阵
    cm = confusion_matrix(y_true_trans, y_pred_trans)
    # 计算每个类别的正确率
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # 创建热力图
    fig = plt.figure(figsize=(5, 4))
    ax = sns.heatmap(cm_percentage, annot=True, cmap='Blues', fmt='.2f', xticklabels=classes, yticklabels=classes,
                     linewidths=0.5, linecolor='black')
    fig.add_axes(ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    fig.tight_layout()
    fig.show()
    fig.savefig(project_path + 'results/figures/cal_interfamily_accuracy.svg')


# 画出损失函数
def fig_loss():
    import os
    current_dir = project_path + 'weights/Proposed/Sig53/impaired/翻转互换128步2头6层多模态/'
    files = os.listdir(current_dir)
    sorted_files = sorted(files, key=lambda x: int(x.split('_')[0]))
    train_loss = []
    test_loss = []
    for file in sorted_files:
        parts = file.split('_')
        if int(parts[0]) >= 48 and int(parts[0]) < 52:
            continue
        if int(parts[0]) > 77:
            break
        train_loss.append(float(parts[3]))
        test_loss.append(float(parts[4]))

    # 画图
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(train_loss, label='train loss')
    ax.plot(test_loss, label='test loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()

    fig.tight_layout()
    fig.show()
    fig.savefig(project_path + 'results/figures/fig_loss.svg')


# 画出不同模型之间的对比结果
def accuracy_compare():

    def pad_trend(data, num=1):
        m, b = np.polyfit(np.arange(num + 1), data[:num + 1], 1)
        tem = np.arange(-num, 0) * m + b
        data = np.insert(data, 0, tem)

        m, b = np.polyfit(np.arange(num + 1), data[-num - 1:], 1)
        tem = np.arange(num + 1, num + 1 + num) * m + b
        data = np.insert(data, len(data), tem)

        return data

    def conv_average(data):
        # 1. 计算原始数据的均值
        original_mean = np.mean(data)

        # 2. 应用移动平均法进行平滑
        N = 7  # 移动平均窗口大小
        # padded_data = np.pad(data, (N // 2, N // 2), mode='edge')
        padded_data = pad_trend(data, num=3)
        smoothed_data = np.convolve(padded_data, np.ones(N) / N, mode='valid')

        # 3. 调整平滑后的数据均值
        smoothed_mean = np.mean(smoothed_data)
        adjusted_smoothed_data = smoothed_data + (original_mean - smoothed_mean)
        return adjusted_smoothed_data


    docPath = './data/'
    names = ['EffNet-B0', 'EffNet-B2', 'EffNet-B4', 'XCiT-Nano', 'XCiT-Tiny12']
    linestyles = [(2, 0),
                  (1, 2),
                  (4, 2, 4, 2),
                  (4, 2, 1, 2),
                  (4, 2, 4, 2, 1, 2)
                  ]
    fig, ax = plt.subplots(figsize=(5, 4))

    # data = np.load(docPath+'58_2024-11-01_21-06_0.564_0.577_0.839_0.749.npy')
    data = np.load(docPath + '57_2024-11-01_16-50_0.565_0.577_0.835_0.750.npy')

    data_smooth = conv_average(data[1])

    # ax.plot(data[0], data[1]*100, label='MTrans')
    line, = ax.plot(data[0], data_smooth * 100, label='Proposed')
    line.set_dashes((4, 2, 1, 2, 1, 2))

    for i, name in enumerate(names):
        data = np.genfromtxt(docPath + name + '.csv', delimiter=',')
        data = data[data[:, 0].argsort()]   # 按时间排序
        data = data[data[:, 0] > -1, :]        # 截断小于-1的值
        line, = ax.plot(data[:, 0], data[:, 1], label=name)
        line.set_dashes(linestyles[i])
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('$E_s/N_0$')
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.show()
    fig.savefig('./figures/accuracy_compare.svg')


def TTA_compare():
    def pad_trend(data, num=1):
        m, b = np.polyfit(np.arange(num + 1), data[:num + 1], 1)
        tem = np.arange(-num, 0) * m + b
        data = np.insert(data, 0, tem)

        m, b = np.polyfit(np.arange(num + 1), data[-num - 1:], 1)
        tem = np.arange(num + 1, num + 1 + num) * m + b
        data = np.insert(data, len(data), tem)

        return data

    def conv_average(data):
        # 1. 计算原始数据的均值
        original_mean = np.mean(data)

        # 2. 应用移动平均法进行平滑
        N = 7  # 移动平均窗口大小
        # padded_data = np.pad(data, (N // 2, N // 2), mode='edge')
        padded_data = pad_trend(data, num=3)
        smoothed_data = np.convolve(padded_data, np.ones(N) / N, mode='valid')

        # 3. 调整平滑后的数据均值
        smoothed_mean = np.mean(smoothed_data)
        adjusted_smoothed_data = smoothed_data + (original_mean - smoothed_mean)
        return adjusted_smoothed_data

    fig, ax = plt.subplots(figsize=(5, 4))

    # data = np.load(docPath+'58_2024-11-01_21-06_0.564_0.577_0.839_0.749.npy')
    data0 = np.load('./data/noTTA_57_2024-11-01_16-50_0.565_0.577_0.835_0.750.npy')
    data1 = np.load('./data/57_2024-11-01_16-50_0.565_0.577_0.835_0.750.npy')

    data0[1] = conv_average(data0[1])
    data1[1] = conv_average(data1[1])

    print('data0_mean = %.4f' % np.mean(data0[1]))
    print('data1_mean = %.4f' % np.mean(data1[1]))

    ax.plot(data0[0], data0[1] * 100, label='Without TTA', linestyle=':', color='b')
    ax.plot(data1[0], data1[1] * 100, label='With TTA', linestyle='--', color='b')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('$E_s/N_0$')
    ax.tick_params(axis='y', labelcolor='b')  # 设置第一个Y轴刻度颜色
    # ax.legend(loc='lower right')
    ax.grid(True)

    # 创建第二个 y 轴
    ax2 = ax.twinx()
    ax2.plot(data0[0], (data1[1]-data0[1])*100, label='Difference', linestyle='-', color='r')
    ax2.set_ylabel('With TTA - Without TTA')  # 设置第二个Y轴标签
    ax2.tick_params(axis='y', labelcolor='r')  # 设置第二个Y轴刻度颜色
    # ax2.legend(loc='lower right')

    fig.legend(bbox_to_anchor=(0.85, 0.75))
    fig.tight_layout()
    fig.show()
    fig.savefig('./figures/TTA_compare.svg')


# 画出平均精度对比的柱状图
def fig_meanAccuracy():
    import numpy as np
    import matplotlib.pyplot as plt

    # 数据
    width = 0.4
    Y = [67.78, 69.73, 71.15, 71.16, 74.99]
    x = np.linspace(0, 4*width, 5)

    # 创建图和轴
    fig, ax = plt.subplots(figsize=(6, 5))

    # 绘制分组柱状图
    for i, y in enumerate(Y):
        ax.bar(width*i, y, width=width)
        ax.text(width*i, y + 0.5, str(y)+'%',
                ha='center', va='bottom', fontsize=10, color='red')  # 标注差值

    # 添加标题、标签和图例
    ax.set_xlabel("Categories", fontsize=12)
    ax.set_ylabel("Accuracy(%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['EffNet-B4\nAugmented', 'EffNet-B4\nOnline', 'XCiT-Tiny12\nAugmented',
                        'XCiT-Tiny12\nOnline', 'Proposed\nAugmented'])

    # 显示
    fig.tight_layout()
    fig.show()
    fig.savefig('./figures/fig_meanAccuracy.svg')


if __name__ == '__main__':
    TTA_compare()
