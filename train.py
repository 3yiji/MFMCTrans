import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import Loader
from utility.utility import fig_accuracy
from config import project_path, sig53_impaired_aug_path, sig53_clean_aug_path
from models import Proposed
import datetime


def main():
    np.set_printoptions(suppress=True)
    path_save = project_path + 'weights/Proposed/Sig53/clean/翻转互换128步2头6层多模态/'
    path_dataset = sig53_clean_aug_path
    path_index_train = project_path + 'index/sig53_clean_aug/index_train.npy'
    path_index_test = project_path + 'index/sig53_clean_aug/index_test.npy'
    cur_epoch = 0
    dB_show = 100
    dB_list = list(range(100, 101, 1))

    epochs = 1000  # 训练的周期数
    batch_size = 256

    max_accuracy = 0
    lr = 0.001
    device = 'cuda'
    # 加载数据迭代器
    dataLoader_train = Loader.DataLoader(path_dataset, path_index_train, batch_size=batch_size, device=device, isPreLoad=True)
    dataLoader_test = Loader.DataLoader(path_dataset, path_index_test, batch_size=batch_size, device=device, isPreLoad=True)

    # 创建模型实例
    model = Proposed.Proposed().to(device=device)
    # model_dict = torch.load(project_path + 'weights/Proposed/Sig53/all/翻转互换128步2头6层多模态2/57_2024-11-01_16-50_0.565_0.577_0.835_0.750.pth')
    # model.load_state_dict(model_dict)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(cur_epoch, epochs):
        total_loss = 0
        model.train()               # 训练模式
        torch.cuda.empty_cache()
        for X, Y in tqdm(dataLoader_train):
            outputs = model(X)
            # 计算损失
            loss = criterion(outputs, Y)

            # 反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            total_loss += loss.item()*Y.shape[0]

        t_loss = total_loss/dataLoader_train.example_n

        # 查看测试的偏差
        model.eval()                # 评估模式
        with torch.no_grad():
            torch.cuda.empty_cache()

            # 计算精度
            _, accuracys, e_loss = fig_accuracy(model, dataLoader_test, dB_list, criterion, batch_size=batch_size*1, isshow=True, aug=4)
            accuracy = accuracys[1][np.where(accuracys[0] == dB_show)][0]
            print(f'epoch {epoch + 1}, Tloss {t_loss:.3f}, Eloss {e_loss:.3f}, Accuracy {accuracy:.3f}, Accuracys_mean {np.mean(accuracys[1]):.3f}')
            print(np.array2string(accuracys, precision=3, max_line_width=400))

            # 保 存模型
            # 获取当前日期和时间
            current_datetime = datetime.datetime.now()
            # 格式化日期和时间为精确到分钟的字符串
            formatted_datetime = current_datetime.strftime("_%Y-%m-%d_%H-%M_")
            path_file = path_save + str(epoch) + formatted_datetime + "%.3f_" % t_loss + "%.3f_" % e_loss + "%.3f_" % accuracy + "%.3f" % np.mean(accuracys[1]) + '.pth'
            torch.save(model.state_dict(), path_file)
            if np.mean(accuracys[1]) > max_accuracy:
                max_accuracy = np.mean(accuracys[1])

    print(f'max accuracy: {max_accuracy:.3f}')


if __name__ == '__main__':
    main()
