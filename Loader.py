import numpy as np
from utility import Myhdf5
import threading
import h5py
import torch


class DataLoader:
    def __init__(self, path_dataset, path_index, batch_size=256, device='cpu', isPreLoad=False):
        self.f = h5py.File(path_dataset, 'r')
        self.index = np.load(path_index)
        self.batch_size = batch_size
        self.device = device
        self.__enable_dataset__ = False
        self.example_n = len(self.index)
        self.max_iterations = int(np.ceil(len(self.index) / self.batch_size))
        self.isPreLoad = isPreLoad
        self.__cachePreData__ = None
        self.__havaCache__ = False
        self.__threading__ = threading.Thread()

    def __iter__(self):
        self.shuffle_index = np.copy(self.index)                          # 复制一个索引，用于打乱排序
        self.shuffle_index_arrange = np.arange(len(self.shuffle_index))         # 顺序的排序
        np.random.shuffle(self.shuffle_index_arrange)                           # 打乱self.shuffle_index_arrange
        self.shuffle_index = self.shuffle_index[self.shuffle_index_arrange]     # 依靠self.shuffle_index_arrange来打乱self.shuffle_index
        self.cur_iter = 0
        return self

    def __next__(self):
        if self.cur_iter < self.max_iterations:
            if self.isPreLoad:
                if self.__threading__.is_alive():                   # 判断是否有线程正在运行，如果有等待运行完成
                    self.__threading__.join()

                if self.__havaCache__:  # 如果存在缓存，直接读取然后开启下一批次的读取
                    x = self.__cachePreData__[0].clone().detach()
                    y = self.__cachePreData__[1].clone().detach()
                    self.__havaCache__ = False
                else:
                    x, y = self.__data_iter__(self.cur_iter)

                if self.cur_iter + 1 < self.max_iterations:         # 预加载数据
                    self.__threading__ = threading.Thread(target=self.__data_iter__,
                                                          args=(self.cur_iter + 1, True))
                    self.__threading__.start()
            else:
                x, y = self.__data_iter__(self.cur_iter)
            self.cur_iter += 1
            return x, y
        else:
            raise StopIteration

    def __data_iter__(self, cur_iter, preload=False):
        current_index = self.shuffle_index_arrange[cur_iter * self.batch_size:(cur_iter + 1) * self.batch_size]
        x = self.__get_data__('X', current_index)
        y = self.__get_data__('Y', current_index)
        x, y = self.__transformer_data__(x, y, device=self.device)
        if preload:
            self.__cachePreData__ = x, y
            self.__havaCache__ = True
        else:
            return x, y

    def __len__(self):
        return self.max_iterations

    # 将从磁盘和内存中加载测试集这两个通道整合起来，便以操作
    def __get_data__(self, dataset, index=None):
        index = np.arange(self.example_n) if index is None else index
        if self.__enable_dataset__:
            if dataset == 'X':
                return self.dataset_x[index]
            elif dataset == 'Y':
                return self.dataset_y[index]
            elif dataset == 'Z':
                return self.dataset_z[index]
            else:
                print('Invalid dataset')
        else:
            data = Myhdf5.read_data(self.f, self.index[index], dataset)
            return data

    # 对数据做变换
    def __transformer_data__(self, X, Y, device='cpu'):

        X = torch.tensor(X, device=device, dtype=torch.float)

        Y = np.argmax(np.array(Y), axis=1)
        Y = torch.tensor(Y, device=device, dtype=torch.long)
        return X, Y

    # 将数据集从磁盘加载保存到内存中，提升速度
    def enable_test_dataset(self):
        self.dataset_x = Myhdf5.read_data(self.f, self.index, 'X')
        self.dataset_y = Myhdf5.read_data(self.f, self.index, 'Y')
        self.dataset_z = Myhdf5.read_data(self.f, self.index, 'Z')
        self.__enable_dataset__ = True


    # 通过信噪比加载数据
    def load_dB(self, snr):
        tem_data_Z = self.__get_data__('Z').squeeze(axis=1)
        index = (tem_data_Z == snr)
        index = np.where(index)[0]
        data_X = self.__get_data__('X', index=index)
        data_Y = self.__get_data__('Y', index=index)
        data_X, data_Y = self.__transformer_data__(data_X, data_Y, device=self.device)
        return data_X, data_Y


    # 通过过信噪比和类别参数加载一个满足条件的数据
    def load_snr_cls_one(self, snr, cls, pos=0):
        tem_data_Z = self.__get_data__('Z').squeeze(axis=1)
        tem_data_Y = np.argmax(self.__get_data__('Y'), axis=1)
        index = np.logical_and(tem_data_Z == snr, tem_data_Y == cls)
        index = np.where(index)[0]
        index = index[:, np.newaxis][pos]
        data_X = self.__get_data__('X', index=index)
        data_Y = self.__get_data__('Y', index=index)
        # data_X, data_Y = self.__transformer_data__(data_X, data_Y, device=self.device)
        return data_X[0], data_Y[0]


    # 加载多个类别的所有数据
    def load_mul_cla(self, clas):
        tem_data_Y = np.argmax(self.__get_data__('Y'), axis=1)
        li_tem = []
        for cla in clas:
            tem = np.where(tem_data_Y == cla)[0]
            li_tem.append(tem)
        index = np.concatenate(li_tem)
        data_X = self.__get_data__('X', index=index)
        data_Y = self.__get_data__('Y', index=index)
        data_X, data_Y = self.__transformer_data__(data_X, data_Y, device=self.device)
        return data_X, data_Y
