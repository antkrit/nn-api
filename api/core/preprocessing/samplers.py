import numpy as np


def train_test_split(*arrays, split=0.7, shuffle=False, seed=None):
    """Split data set into random train and test subset.

    :param arrays: arrays to split
    :param split: number between 0 and 1, proportion between train and test
        subsets, defaults to 0.7
    :param shuffle: specifies whether to shuffle the data, defaults to False
    :param seed: random seed to reproduce random results, defaults to None
    :raises ValueError: if less than one array is passed to the function
    :return: partitioned arrays
    """
    if len(arrays) == 0:
        raise ValueError(f'Cannot split 0 arrays.')

    size = len(arrays[0])
    idx = np.arange(size)

    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(idx)

    split_idx = int(size * split)

    return [
        b
        for a in arrays
        for b in (a[idx][:split_idx], a[idx][split_idx:])
    ]
