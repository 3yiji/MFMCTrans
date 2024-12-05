import numpy as np


def gene_index():
    n_train = int(5.3e6*4)
    n_test = int(106e3*4)
    index_train = np.arange(0, n_train)
    index_test = np.arange(n_train, n_train+n_test)
    np.save('index_train.npy', index_train)
    np.save('index_test.npy', index_test)


if __name__ == '__main__':
    pass
