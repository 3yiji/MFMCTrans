import h5py
import numpy as np
from config import project_path, sig53_clean_path, sig53_clean_aug_path
from tqdm import tqdm


def gene_index():
    n_train = int(1.06e6)
    n_test = int(10.6e3)
    index_train = np.arange(0, n_train)
    index_test = np.arange(n_train, n_train+n_test)
    np.save('index_train.npy', index_train)
    np.save('index_test.npy', index_test)


# 生成增强数据集
def gen_sig_aug():
    file_path_r = sig53_clean_path
    file_path_w = sig53_clean_aug_path
    num_repeats = 4
    fr = h5py.File(file_path_r, 'r')
    fr_shape = fr['X'].shape
    fw = h5py.File(file_path_w, 'w')
    fw.create_dataset('Y', shape=(fr['Y'].shape[0]*num_repeats, fr['Y'].shape[1]), dtype=np.int16)
    fw.create_dataset('Z', shape=(fr['Z'].shape[0]*num_repeats, fr['Z'].shape[1]), dtype=np.int16)
    fw.create_dataset('X', shape=(fr['X'].shape[0]*num_repeats, fr['X'].shape[1], fr['X'].shape[2]), dtype=np.float32)

    batch_size = 500
    for i in tqdm(range(0, fr_shape[0], batch_size)):
        x_tem = fr['X'][i:i + batch_size]

        # 翻转互换
        x_tem = [x_tem, x_tem[:, :, ::-1], x_tem[:, ::-1], x_tem[:, ::-1, ::-1]]

        # 使用 np.stack 进行交错合并
        x_tem = np.stack(x_tem, axis=1)
        # 将 result reshape 成目标形状
        x_tem = x_tem.reshape(-1, x_tem.shape[2], x_tem.shape[3])

        fw['Y'][i * num_repeats: (i + batch_size) * num_repeats] = np.repeat(fr['Y'][i:i + batch_size], num_repeats, axis=0)
        fw['Z'][i * num_repeats: (i + batch_size) * num_repeats] = np.repeat(fr['Z'][i:i + batch_size], num_repeats, axis=0)
        fw['X'][i*num_repeats: (i + batch_size)*num_repeats] = x_tem

    fr.close()
    fw.close()


if __name__ == '__main__':
    gen_sig_aug()


