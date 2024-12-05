import h5py
import numpy as np
from multiprocessing import Pool

def loader_batch(f, index, dataset, batch=1000):
    for i in range(0, len(index), batch):
        ind = index[i:i+batch]     # 原始的索引
        order1 = np.argsort(ind)  # 可以排序索引的下标
        sorted_index = ind[order1]  # 排列后的索引
        sorted_data = f[dataset][sorted_index]  # 排列后的数据
        origin_data = sorted_data[np.argsort(order1)]   # 原始的数据
        yield ind, origin_data


def write_data(f, data, index, dataset):
    order1 = np.argsort(index)  # 可以排序索引的下标
    sorted_index = index[order1]  # 排列后的索引
    sorted_data = data[order1]  # 排列后的数据
    f[dataset][sorted_index] = sorted_data


def read_data(f, index, dataset):
    datax = []
    batch_size = 1000
    order1 = np.argsort(index)  # 可以排序索引的下标
    sorted_index = index[order1]  # 排列后的索引

    for ind in range(0, len(sorted_index), batch_size):
        h5datax = f[dataset][sorted_index[ind:ind + batch_size]]  # 排列后的数据
        datax.append(h5datax)
    sorted_X = np.concatenate(datax, axis=0)  # 排列后的数据

    origin_X = sorted_X[np.argsort(order1)]  # 恢复原始排序的数据
    return origin_X

