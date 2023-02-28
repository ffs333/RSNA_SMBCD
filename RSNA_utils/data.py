import random

import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .get_augs import get_transforms, line_dropout, circle_aug


def prepare_loaders(_cfg, folds, fold):
    """
    Prepare and build train and eval data loaders
    """

    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    if _cfg.pos_multiplier > 1:
        pos_df = train_folds.loc[train_folds[(train_folds.cancer == 1) & (train_folds.fold != 999)].index.repeat(_cfg.pos_multiplier)]
        train_folds = pd.concat([train_folds, pos_df]).reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)

    _cfg.neg_samples = len(train_folds[train_folds.cancer == 0])
    _cfg.pos_samples = len(train_folds[train_folds.cancer == 1])
    print(f'Size of train dataset: {len(train_folds)}  \n'
          f'class 0: {len(train_folds[train_folds.cancer==0])} | class 1: {len(train_folds[train_folds.cancer==1])}')
    print(f'Size of valid dataset: {len(valid_folds)} \n'
          f'class 0: {len(valid_folds[valid_folds.cancer==0])} | class 1: {len(valid_folds[valid_folds.cancer==1])}')

    transforms_0, transforms_1, valid_transforms = get_transforms(_cfg)

    if _cfg.dataset == 'v1':
        train_dataset = DatasetV1(_cfg, train_folds, use_meta=_cfg.use_meta, train_mode=True,
                                  transforms=[transforms_0, transforms_1, valid_transforms])

        valid_dataset = DatasetV1(_cfg, valid_folds, use_meta=_cfg.use_meta, train_mode=False,
                                  transforms=valid_transforms)

        collate = None
        valid_bs = _cfg.valid_bs

    elif _cfg.dataset == 'v2':
        train_dataset = DatasetV2PatientTrain(_cfg, train_folds,
                                              transforms=[transforms_0, transforms_1, valid_transforms])

        valid_dataset = DatasetV2PatientValid(valid_folds, transforms=valid_transforms)

        collate = collate_fn
        valid_bs = _cfg.valid_bs * 2

    elif _cfg.dataset == 'transformer':
        train_dataset = DatasetTransformer(_cfg, train_folds, image_size=_cfg.image_size, train_mode=True,
                                           transforms=[transforms_0, transforms_1, valid_transforms])

        valid_dataset = DatasetTransformer(_cfg, valid_folds, image_size=_cfg.image_size, train_mode=False,
                                           transforms=valid_transforms)

        collate = None
        valid_bs = _cfg.valid_bs

    elif _cfg.dataset == 'stripes':
        train_dataset = DatasetStripes(_cfg, train_folds, use_meta=_cfg.use_meta, train_mode=True,
                                       transforms=[transforms_0, transforms_1, valid_transforms])

        valid_dataset = DatasetStripes(_cfg, valid_folds, use_meta=_cfg.use_meta, train_mode=False,
                                       transforms=valid_transforms)

        collate = None
        valid_bs = _cfg.valid_bs

    elif _cfg.dataset == 'transformer_stripes':
        train_dataset = DatasetTransformerStripes(_cfg, train_folds, image_size=_cfg.image_size, train_mode=True,
                                                  transforms=[transforms_0, transforms_1, valid_transforms])

        valid_dataset = DatasetTransformerStripes(_cfg, valid_folds, image_size=_cfg.image_size, train_mode=False,
                                                  transforms=valid_transforms)

        collate = None
        valid_bs = _cfg.valid_bs

    elif _cfg.dataset == 'transformer_stripes_MIP':
        train_dataset = DatasetTransformerStripesMIP(_cfg, train_folds, image_size=_cfg.image_size, train_mode=True,
                                                     transforms=[transforms_0, transforms_1, valid_transforms])

        valid_dataset = DatasetTransformerStripesMIP(_cfg, valid_folds, image_size=_cfg.image_size, train_mode=False,
                                                     transforms=valid_transforms)

        collate = collate_MIP_fn
        valid_bs = _cfg.valid_bs

    elif _cfg.dataset == 'attention':
        train_dataset = DatasetV1Atten(_cfg, train_folds, train_mode=True,
                                       transforms=[transforms_0, transforms_1, valid_transforms])

        valid_dataset = DatasetV1Atten(_cfg, valid_folds, train_mode=False, transforms=valid_transforms)

        collate = collate_fn_att
        valid_bs = _cfg.valid_bs

    else:
        raise ValueError('Error in "prepare_loaders" function:',
                         f'Wrong dataset name. Choose one from ["v1", "v2", "transformer", "stripes"] ')

    train_loader = DataLoader(train_dataset,
                              batch_size=_cfg.train_bs,
                              shuffle=True,
                              collate_fn=collate,
                              num_workers=_cfg.num_workers, pin_memory=True, drop_last=True)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=valid_bs,
                              shuffle=False,
                              collate_fn=collate,
                              num_workers=_cfg.num_workers, pin_memory=False, drop_last=False)

    return train_loader, valid_loader, valid_folds[_cfg.interesing_cols]


def prepare_loaders_fulltrain(_cfg, folds):
    """
    Prepare and build train and eval data loaders
    """

    if _cfg.pos_multiplier > 1:
        pos_df = folds.loc[folds[(folds.cancer == 1) & (folds.fold != 999)].index.repeat(_cfg.pos_multiplier)]
        folds = pd.concat([folds, pos_df]).reset_index(drop=True)

    _cfg.neg_samples = len(folds[folds.cancer == 0])
    _cfg.pos_samples = len(folds[folds.cancer == 1])

    print(f'Size of train dataset: {len(folds)}  \n'
          f'class 0: {len(folds[folds.cancer==0])} | class 1: {len(folds[folds.cancer==1])}')

    transforms_0, transforms_1, valid_transforms = get_transforms(_cfg)

    if _cfg.dataset == 'v1':
        train_dataset = DatasetV1(_cfg, folds, use_meta=_cfg.use_meta, train_mode=True,
                                  transforms=[transforms_0, transforms_1, valid_transforms])

    elif _cfg.dataset == 'stripes':
        train_dataset = DatasetStripes(_cfg, folds, use_meta=_cfg.use_meta, train_mode=True,
                                       transforms=[transforms_0, transforms_1, valid_transforms])

    elif _cfg.dataset == 'transformer_stripes':
        train_dataset = DatasetTransformerStripes(_cfg, folds, image_size=_cfg.image_size, train_mode=True,
                                                  transforms=[transforms_0, transforms_1, valid_transforms])

    else:
        raise ValueError('Error in "prepare_loaders" function:',
                         f'Wrong dataset name. Choose one from ["v1", "v2"] ')

    train_loader = DataLoader(train_dataset,
                              batch_size=_cfg.train_bs,
                              shuffle=True,
                              num_workers=_cfg.num_workers, pin_memory=True, drop_last=True)

    return train_loader, folds


class DatasetV1(torch.utils.data.Dataset):
    def __init__(self, cfg, data, use_meta=True, train_mode=True, transforms=None):
        self.data = data.copy()
        self.use_meta = use_meta
        self.train_mode = train_mode
        self.use_aug_prob = cfg.use_aug_prob
        if self.train_mode:

            self.transforms_0 = transforms[0]
            self.transforms_1 = transforms[1]
            self.transforms = transforms[2]
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.loc[index]
        img = cv2.imread(sample['path'])

        if sample['laterality'] == 'L':
            img = np.flip(img, axis=1)

        img = img.transpose(2, 0, 1)

        label = sample['cancer']
        if self.train_mode:
            if random.random() < self.use_aug_prob:
                img = self.transforms_1(img) if label == 1 else self.transforms_0(img)
            else:
                img = self.transforms(img)
        else:
            img = self.transforms(img)

        difficult = sample['difficult_negative_case'].astype(int)

        if self.use_meta:
            view = ohe_encoding_view(sample['view'])
            age = np.array([sample['age']])
            implant = np.array([sample['implant']])

            meta = np.concatenate([view, age, implant]).astype(np.float32)

            return img, meta, label, difficult

        return img, label, difficult


def ohe_encoding_view(x):
    out = np.zeros(3)
    if x == 'CC':
        out[0] = 1
    elif x == 'MLO':
        out[1] = 1
    else:
        out[2] = 1
    return out


class DatasetV2PatientTrain(torch.utils.data.Dataset):
    def __init__(self, cfg, data, transforms=None):
        self.data = data.copy()
        self.use_aug_prob = cfg.use_aug_prob

        self.transforms_0 = transforms[0]
        self.transforms_1 = transforms[1]
        self.transforms = transforms[2]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index, 1].reset_index(drop=True)

        indexes = sample.index.values.copy()
        np.random.shuffle(indexes)

        indexes = indexes[:3]
        if len(indexes) < 3:
            indexes = np.append(indexes, np.random.choice(indexes, 3 - len(indexes)))

        sample = sample.loc[indexes].reset_index(drop=True)

        full_img = []
        for i in range(len(sample)):
            img = cv2.imread(sample.loc[i, 'path'], cv2.IMREAD_GRAYSCALE)
            if sample.loc[i, 'laterality'] == 'L':
                img = np.flip(img, axis=1)
            full_img.append(img)

        full_img = np.stack(full_img)

        label = int(sample['cancer'].mean() > 0.5)

        if random.random() < self.use_aug_prob:
            full_img = self.transforms_1(full_img) if label == 1 else self.transforms_0(full_img)
        else:
            full_img = self.transforms(full_img)

        ext = (sample['fold'] == 999).astype(int)

        return full_img, label, ext


class DatasetV2PatientValid(torch.utils.data.Dataset):
    def __init__(self, data, transforms=None):
        self.data = data.copy()

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index, 1].reset_index(drop=True)

        if len(sample) < 3:
            indexes = sample.index.values.copy()
            indexes = np.append(indexes, np.random.choice(indexes, 3 - len(indexes)))
            sample = sample.loc[indexes].reset_index(drop=True)

        full_img = []
        for i in range(len(sample)):
            img = cv2.imread(sample.loc[i, 'path'], cv2.IMREAD_GRAYSCALE)
            if sample.loc[i, 'laterality'] == 'L':
                img = np.flip(img, axis=1)
            full_img.append(img)

        full_img = np.stack(full_img)

        label = int(sample['cancer'].mean() > 0.5)

        full_img = self.transforms(full_img)

        ext = (sample['fold'] == 999).astype(int)

        return full_img, label, ext


def collate_fn(batch):
    """
    collate function is necessary for transferring data into GPU
    :param batch: Input tensor
    :return tuple with labels and batch tensors
    """
    img = torch.cat([x[0] for x in batch], dim=0)
    lengths = torch.tensor([len(x[0]) for x in batch])

    label = torch.tensor([x[1] for x in batch], dtype=torch.float32)
    difficult = torch.tensor([x[2] for x in batch])

    return img, lengths, label, difficult


class DatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, cfg, data, image_size=(1536, 1024), train_mode=True, transforms=None):
        self.data = data.copy()
        self.train_mode = train_mode
        self.use_aug_prob = cfg.use_aug_prob

        self.size_h = image_size[0]
        self.size_w = image_size[1]
        self.sc = self.size_h / self.size_w

        if self.train_mode:

            self.transforms_0 = transforms[0]
            self.transforms_1 = transforms[1]
            self.transforms = transforms[2]
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.data)

    @staticmethod
    def fit_image(x, more=20):

        # regions of non-empty pixels
        output = cv2.connectedComponentsWithStats((x[:, :, 0] > more).astype(np.uint8)[:, :], 8, cv2.CV_32S)
        stats = output[2]

        idx = stats[1:, 4].argmax() + 1
        x1, y1, w, h = stats[idx][:4]
        x2 = x1 + w
        y2 = y1 + h

        # cutting out the breast data
        x_fit = x[y1: y2, x1: x2]
        return x_fit

    def padding(self, array):

        h = array.shape[0]
        w = array.shape[1]

        if h / w > self.sc:
            need = int(h / self.sc)
            b = (need - w) // 2
            bb = need - b - w
            a = aa = 0
        elif h / w < self.sc:
            need = int(w * self.sc)
            a = (need - h) // 2
            aa = need - a - h
            b = bb = 0
        else:
            a = aa = b = bb = 0

        return np.pad(array, pad_width=((a, aa), (b, bb), (0, 0)), mode='constant')

    def __getitem__(self, index):
        sample = self.data.loc[index]
        img = cv2.imread(sample['path'])
        img = self.fit_image(img)
        img = self.padding(img)
        img = cv2.resize(img, (self.size_w, self.size_h))

        if sample['laterality'] == 'L':
            img = np.flip(img, axis=1)

        img = img.transpose(2, 0, 1)

        label = sample['cancer']
        if self.train_mode:
            if random.random() < self.use_aug_prob:
                img = self.transforms_1(img) if label == 1 else self.transforms_0(img)
            else:
                img = self.transforms(img)
        else:
            img = self.transforms(img)

        ext = (sample['fold'] == 999).astype(int)

        return img, label, ext


class DatasetStripes(torch.utils.data.Dataset):
    def __init__(self, cfg, data, use_meta=True, train_mode=True, transforms=None):
        self.data = data.copy()
        self.use_meta = use_meta
        self.train_mode = train_mode
        self.use_aug_prob = cfg.use_aug_prob
        if self.train_mode:

            self.transforms_0 = transforms[0]
            self.transforms_1 = transforms[1]
            self.transforms = transforms[2]
        else:
            self.transforms = transforms

        self.epoch = 0
        self.gap_dict = cfg.gap_dict
        self.last_val = self.gap_dict[sorted(list(self.gap_dict.keys()))[-1]]
        self.first_gap = self.gap_dict[sorted(list(self.gap_dict.keys()))[0]] + 1
        self.stripe_prob = cfg.stripe_prob
        self.circle_aug_prob = cfg.circle_aug_prob
        self.img_size = cfg.img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.loc[index]
        img = cv2.imread(sample['path'])
        label = sample['cancer']

        # if self.train_mode:
        #    if random.random() < self.circle_aug_prob:
        #        img = img if label == 1 else circle_aug(img, IMG_SIZE=self.img_size, min_c=1, max_c=4)

        #if sample['laterality'] == 'L':
        #    img = np.flip(img, axis=1)
                
        img = img.transpose(2, 0, 1)

        if self.train_mode:
            if random.random() < self.use_aug_prob:
                img = self.transforms_1(img) if label == 1 else self.transforms_0(img)
            else:
                img = self.transforms(img)

            if random.random() < self.stripe_prob:
                im = img.cpu().numpy()
                gap = np.random.randint(self.gap_dict.get(self.epoch, self.last_val), self.first_gap)

                mask = line_dropout(np.ones_like(im[0]) * 255,
                                    gap=gap, err=100,
                                    line=True,
                                    square=False)
                mask = torch.FloatTensor(mask)
                img = torch.where(mask < 255, 0., img)
        else:
            img = self.transforms(img)

        ext = (sample['fold'] == 999).astype(int)

        return img, label, ext


class DatasetTransformerStripes(torch.utils.data.Dataset):
    def __init__(self, cfg, data, image_size=(1536, 1024), train_mode=True, transforms=None):
        self.data = data.copy()
        self.train_mode = train_mode
        self.use_aug_prob = cfg.use_aug_prob

        self.size_h = image_size[0]
        self.size_w = image_size[1]
        self.sc = self.size_h / self.size_w

        if self.train_mode:

            self.transforms_0 = transforms[0]
            self.transforms_1 = transforms[1]
            self.transforms = transforms[2]
        else:
            self.transforms = transforms

        self.epoch = 0
        self.gap_dict = cfg.gap_dict
        self.last_val = self.gap_dict[sorted(list(self.gap_dict.keys()))[-1]]
        self.stripe_prob = cfg.stripe_prob
        self.circle_aug_prob = cfg.circle_aug_prob
        self.img_size = cfg.img_size

    def __len__(self):
        return len(self.data)

    @staticmethod
    def fit_image(x, more=20):

        # regions of non-empty pixels
        output = cv2.connectedComponentsWithStats((x[:, :, 0] > more).astype(np.uint8)[:, :], 8, cv2.CV_32S)
        stats = output[2]

        idx = stats[1:, 4].argmax() + 1
        x1, y1, w, h = stats[idx][:4]
        x2 = x1 + w
        y2 = y1 + h

        # cutting out the breast data
        x_fit = x[y1: y2, x1: x2]
        return x_fit

    def padding(self, array):

        h = array.shape[0]
        w = array.shape[1]

        if h / w > self.sc:
            need = int(h / self.sc)
            b = (need - w) // 2
            bb = need - b - w
            a = aa = 0
        elif h / w < self.sc:
            need = int(w * self.sc)
            a = (need - h) // 2
            aa = need - a - h
            b = bb = 0
        else:
            a = aa = b = bb = 0

        return np.pad(array, pad_width=((a, aa), (b, bb), (0, 0)), mode='constant')

    def __getitem__(self, index):
        sample = self.data.loc[index]
        img = cv2.imread(sample['path'])
        label = sample['cancer']

        if self.train_mode:
            if random.random() < self.circle_aug_prob:
                img = img if label == 1 else circle_aug(img, IMG_SIZE=self.img_size, min_c=1, max_c=4)

        img = self.fit_image(img)
        img = self.padding(img)
        img = cv2.resize(img, (self.size_w, self.size_h))

        if sample['laterality'] == 'L':
            img = np.flip(img, axis=1)

        img = img.transpose(2, 0, 1)

        if self.train_mode:
            if random.random() < self.use_aug_prob:
                img = self.transforms_1(img) if label == 1 else self.transforms_0(img)
            else:
                img = self.transforms(img)

            if random.random() < self.stripe_prob:
                im = img.cpu().numpy()
                gap = self.gap_dict.get(self.epoch, self.last_val)

                mask = line_dropout(np.ones_like(im[0]) * 255,
                                    gap=gap, err=100,
                                    line=True,
                                    square=False)
                mask = torch.FloatTensor(mask)
                img = torch.where(mask < 255, 0., img)
        else:
            img = self.transforms(img)

        ext = (sample['fold'] == 999).astype(int)

        return img, label, ext


class DatasetTransformerStripesMIP(torch.utils.data.Dataset):
    def __init__(self, cfg, data, image_size=(1536, 1024), train_mode=True, transforms=None):
        self.data = data.copy()
        self.train_mode = train_mode
        self.use_aug_prob = cfg.use_aug_prob

        self.size_h = image_size[0]
        self.size_w = image_size[1]
        self.sc = self.size_h / self.size_w

        if self.train_mode:

            self.transforms_0 = transforms[0]
            self.transforms_1 = transforms[1]
            self.transforms = transforms[2]
        else:
            self.transforms = transforms

        self.epoch = 0
        self.gap_dict = cfg.gap_dict
        self.last_val = self.gap_dict[sorted(list(self.gap_dict.keys()))[-1]]
        self.stripe_prob = cfg.stripe_prob

    def __len__(self):
        return len(self.data)

    @staticmethod
    def fit_image(x, more=20):

        # regions of non-empty pixels
        output = cv2.connectedComponentsWithStats((x[:, :, 0] > more).astype(np.uint8)[:, :], 8, cv2.CV_32S)
        stats = output[2]

        idx = stats[1:, 4].argmax() + 1
        x1, y1, w, h = stats[idx][:4]
        x2 = x1 + w
        y2 = y1 + h

        # cutting out the breast data
        x_fit = x[y1: y2, x1: x2]
        return x_fit

    def padding(self, array):

        h = array.shape[0]
        w = array.shape[1]

        if h / w > self.sc:
            need = int(h / self.sc)
            b = (need - w) // 2
            bb = need - b - w
            a = aa = 0
        elif h / w < self.sc:
            need = int(w * self.sc)
            a = (need - h) // 2
            aa = need - a - h
            b = bb = 0
        else:
            a = aa = b = bb = 0

        return np.pad(array, pad_width=((a, aa), (b, bb), (0, 0)), mode='constant')

    def __getitem__(self, index):
        sample = self.data.loc[index]
        img = cv2.imread(sample['path'])
        img = self.fit_image(img)
        img = self.padding(img)
        img = cv2.resize(img, (self.size_w, self.size_h))

        if sample['laterality'] == 'L':
            img = np.flip(img, axis=1)

        img = img.transpose(2, 0, 1)

        label = sample['cancer']
        if self.train_mode:
            if random.random() < self.use_aug_prob:
                img = self.transforms_1(img) if label == 1 else self.transforms_0(img)
            else:
                img = self.transforms(img)

            if random.random() < self.stripe_prob:
                im = img.cpu().numpy()
                gap = self.gap_dict.get(self.epoch, self.last_val)

                mask = line_dropout(np.ones_like(im[0]) * 255,
                                    gap=gap, err=100,
                                    line=True,
                                    square=False)
                mask = torch.FloatTensor(mask)
                img = torch.where(mask < 255, 0., img)
        else:
            img = self.transforms(img)

        ext = (sample['fold'] == 999).astype(int)
        img = img.unfold(1, 256, 256).unfold(2, 256, 256)
        C, p1, p2, H, W = img.shape
        img = img.permute(1, 2, 0, 3, 4).reshape(p1 * p2, C, H, W)
        chosen = [idx for idx in range(img.size(0)) if img[idx].mean() > 0.03]

        return img[chosen], label, ext, len(chosen)


def collate_MIP_fn(batch):
    """
    collate function is necessary for transferring data into GPU
    :param batch: Input tensor
    :return tuple with labels and batch tensors
    """
    img = torch.cat([x[0] for x in batch], dim=0)

    label = torch.tensor([x[1] for x in batch], dtype=torch.float32)
    difficult = torch.tensor([x[2] for x in batch])
    lens = torch.tensor([x[3] for x in batch])

    return img, label, difficult, lens


class DatasetV1Atten(torch.utils.data.Dataset):
    def __init__(self, cfg, data, train_mode=True, transforms=None):
        self.data = data.copy()
        self.use_aug_prob = cfg.use_aug_prob
        self.train_mode = train_mode

        if self.train_mode:
            self.transforms_0 = transforms[0]
            self.transforms_1 = transforms[1]
            self.transforms = transforms[2]
        else:
            self.transforms = transforms

        self.epoch = 0
        self.gap_dict = cfg.gap_dict
        self.last_val = self.gap_dict[sorted(list(self.gap_dict.keys()))[-1]]
        self.first_gap = self.gap_dict[sorted(list(self.gap_dict.keys()))[0]] + 1
        self.stripe_prob = cfg.stripe_prob
        # self.circle_aug_prob = cfg.circle_aug_prob
        # self.img_size = cfg.img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index, 1].reset_index(drop=True)

        indexes = sample.index.values.copy()
        np.random.shuffle(indexes)

        if self.train_mode:
            indexes = indexes[:4]
            sample = sample.loc[indexes].reset_index(drop=True)
        label = int(sample['cancer'].mean() > 0.5)

        full_img = []
        for i in range(len(sample)):
            img = cv2.imread(sample.loc[i, 'path'])
            if sample.loc[i, 'laterality'] == 'L':
                img = np.flip(img, axis=1)
            img = img.transpose(2, 0, 1)
            if self.train_mode:
                if random.random() < self.use_aug_prob:
                    img = self.transforms_1(img) if label == 1 else self.transforms_0(img)
                else:
                    img = self.transforms(img)

                if random.random() < self.stripe_prob:
                    im = img.cpu().numpy()
                    gap = np.random.randint(self.gap_dict.get(self.epoch, self.last_val), self.first_gap)

                    mask = line_dropout(np.ones_like(im[0]) * 255,
                                        gap=gap, err=100,
                                        line=True,
                                        square=False)
                    mask = torch.FloatTensor(mask)
                    img = torch.where(mask < 255, 0., img)
            else:
                img = self.transforms(img)

            full_img.append(img)

        return full_img, label


def collate_fn_att(batch):
    """
    collate function is necessary for transferring data into GPU
    :param batch: Input tensor
    :return tuple with labels and batch tensors
    """
    img = torch.cat([torch.stack(x[0]) for x in batch], dim=0)
    lengths = torch.tensor([len(x[0]) for x in batch])

    label = torch.tensor([x[1] for x in batch], dtype=torch.float32)

    return img, lengths, label
