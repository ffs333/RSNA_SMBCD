import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt


def get_logger(filename):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def set_seed(seed=42):
    """
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # was commented
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))


def make_dist_plot(df0):
    df0 = df0.copy()
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    axs[0].hist(df0['prediction'].values, bins=100)
    axs[0].set_title('full', fontsize=30)
    axs[1].hist(df0[df0.cancer == 1]['prediction'].values, bins=100)
    axs[1].set_title('cancer=1', fontsize=30)
    axs[2].hist(df0[(df0.cancer == 0) & (df0.prediction > 0.012)]['prediction'].values, bins=100)
    axs[2].set_title('cancer=0 | pred > 0.012', fontsize=30)
    plt.show()
