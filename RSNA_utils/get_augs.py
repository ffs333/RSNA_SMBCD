import math
import random

import cv2
import torch
import numpy as np
import monai.transforms as T
from PIL import Image, ImageDraw


def get_transforms(_cfg):

    transforms_0 = T.Compose([
        # Main
        T.Lambda(func=lambda x: x / 255.),
        # T.NormalizeIntensity(subtrahend=125, divisor=125),

        T.RandFlip(prob=_cfg.flip_prob_0, spatial_axis=1),

        T.RandZoom(prob=_cfg.zoom_prob_0, min_zoom=0.85, max_zoom=1.25),
        T.RandRotate(prob=_cfg.rotate_prob_0, range_x=0.4),

        T.OneOf([
                T.Rand2DElastic(spacing=(_cfg.img_size//8, _cfg.img_size//8),
                                prob=_cfg.elastic_prob_0, magnitude_range=(1, 2), padding_mode='zeros'),
                T.RandAffine(prob=_cfg.affine_prob_0, rotate_range=None, shear_range=(0.05, 0.2))]),

        T.RandAdjustContrast(prob=_cfg.contr_prob_0, gamma=(0.9, 1.1)),
        T.RandCoarseDropout(holes=2, spatial_size=_cfg.img_size//60, max_spatial_size=_cfg.img_size//40,
                            fill_value=None, max_holes=5, prob=_cfg.drop_prob_0),
        T.EnsureType(dtype=torch.float32)
    ])

    transforms_1 = T.Compose([
        # Main
        T.Lambda(func=lambda x: x / 255.),
        # T.NormalizeIntensity(subtrahend=125, divisor=125),
        T.RandFlip(prob=_cfg.flip_prob_0, spatial_axis=1),

        T.RandZoom(prob=_cfg.zoom_prob_1, min_zoom=0.85, max_zoom=1.25),
        T.RandRotate(prob=_cfg.rotate_prob_1, range_x=0.4),
        T.OneOf([
                T.Rand2DElastic(spacing=(_cfg.img_size//8, _cfg.img_size//8),
                                prob=_cfg.elastic_prob_1, magnitude_range=(1, 2), padding_mode='zeros'),
                T.RandAffine(prob=_cfg.affine_prob_1, rotate_range=None, shear_range=(0.05, 0.2))]),

        T.RandAdjustContrast(prob=_cfg.contr_prob_1, gamma=(0.9, 1.1)),
        T.RandCoarseDropout(holes=2, spatial_size=_cfg.img_size//60, max_spatial_size=_cfg.img_size//40,
                            fill_value=None, max_holes=5, prob=_cfg.drop_prob_1),
        T.EnsureType(dtype=torch.float32)
    ])

    valid_transforms = T.Compose([
        # Main
        T.Lambda(func=lambda x: x / 255.),
        # T.NormalizeIntensity(subtrahend=125, divisor=125),
        T.EnsureType(dtype=torch.float32)
    ])

    if 'stripes' in _cfg.dataset and False:
        transforms_0.transforms = [x for x in transforms_0.transforms if not isinstance(x, T.RandCoarseDropout)]
        transforms_1.transforms = [x for x in transforms_1.transforms if not isinstance(x, T.RandCoarseDropout)]
        print(f'Coarse dropout augmenations has dropped because of line dropout dataset!')
    return transforms_0, transforms_1, valid_transforms


'''# Randzoom
                train_loader.dataset.transforms_0.transforms[2].prob = 0.35
                train_loader.dataset.transforms_1.transforms[2].prob = 0.35

                # Rotate
                train_loader.dataset.transforms_0.transforms[3].prob = 0.35
                train_loader.dataset.transforms_1.transforms[3].prob = 0.35

                # OneOf Elastic Affine
                train_loader.dataset.transforms_0.transforms[4].transforms[0].prob = 0.35
                train_loader.dataset.transforms_0.transforms[4].transforms[1].prob = 0.35
                train_loader.dataset.transforms_1.transforms[4].transforms[0].prob = 0.35
                train_loader.dataset.transforms_1.transforms[4].transforms[1].prob = 0.35

                # Contrast
                train_loader.dataset.transforms_0.transforms[5].prob = 0.4
                train_loader.dataset.transforms_1.transforms[5].prob = 0.4

                # Coarse Dropout
                if 'stripes' not in CFG.dataset:
                    train_loader.dataset.transforms_0.transforms[6].prob = 0.35
                    train_loader.dataset.transforms_1.transforms[6].prob = 0.35
                else:
                    train_loader.dataset.stripe_prob = 0.75'''


def line_dropout(img, gap: int, err: int, line: bool, square: bool):
    img = Image.fromarray(img.astype(np.uint8))
    # initiation data/var
    draw = ImageDraw.Draw(img)
    width, height = img.size

    class Coord:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    # calculate endpoint with angle/length
    def calcEnd(x, y, angle, length):
        endx = int(x - (math.cos(math.radians(angle)) * length))
        endy = int(y - (math.sin(math.radians(angle)) * length))
        return endx, endy

    if line:
        currx = 0
        curry = 0
        coordlist = []

    # generate the set of points
    while currx <= width:
        while curry <= height:
            coordlist.append(Coord(currx + random.randint(0, err), curry + random.randint(0, err)))
            curry += gap
        curry = gap
        currx += gap
    # draw line with random angle/length
    for c in coordlist:
        length = random.randint(10, 400)
        randangle = random.randint(0, 359)
        endx, endy = calcEnd(c.x, c.y, randangle, length)
        draw.line((c.x, c.y, endx, endy), fill=0,  # random.randint(0,3),#'black',
                  width=random.randint(1, 4))
    if square:
        currx = 0
        curry = 0
        coordlist = []

    # generate the set of points
    while currx <= width:
        while curry <= height:
            coordlist.append(Coord(currx + random.randint(0, err), curry + random.randint(0, err)))
            curry += gap
        curry = gap
        currx += gap
    # draw square with random angle/length
    for c in coordlist:
        length = random.randint(5, 15)
        randangle = random.randint(0, 359)
        endx, endy = calcEnd(c.x, c.y, randangle, length)
        draw.line((c.x, c.y, endx, endy), fill=0,  # random.randint(0,3),#'black',
                  width=random.randint(5, 15))

    return np.array(img)


def circle_aug(img, IMG_SIZE=2048, min_c=1, max_c=4):
    how_much = np.random.randint(min_c, max_c)
    existed_cntr = []

    for _ in range(how_much):
        img_ = img.copy()
        center_coor = (np.random.randint(IMG_SIZE // 4, IMG_SIZE - (IMG_SIZE // 4)),
                       np.random.randint(IMG_SIZE // 4, IMG_SIZE - (IMG_SIZE // 4)))
        not_per = False
        cnt = 0
        while (img_[center_coor[0] - 20:center_coor[0] + 20,
               center_coor[1] - 20:center_coor[1] + 20] == 0).sum() > 1200 or not not_per:
            cnt += 1
            not_per = False
            center_coor = (np.random.randint(IMG_SIZE // 4, IMG_SIZE - (IMG_SIZE // 4)),
                           np.random.randint(IMG_SIZE // 4, IMG_SIZE - (IMG_SIZE // 4)))
            if sum([np.abs(np.array(x) - center_coor).min() < 100 for x in existed_cntr]) == 0 or len(
                    existed_cntr) == 0:
                not_per = True
            if cnt > 100:
                break
        existed_cntr.append(center_coor)
        radius = np.random.randint(65, 90)

        c_val = np.random.randint(10, 45)
        color = (c_val, c_val, c_val)
        thickness = np.random.randint(14, 18)

        angle = 0
        startAngle = 0
        endAngle = 360
        axesLength = (radius + np.random.randint(-10, 10), radius + np.random.randint(-10, 10))
        img_ = cv2.ellipse(img_, center_coor, axesLength,
                           angle, startAngle, endAngle, color, thickness)

        c_val = np.random.randint(120, 180)
        color = (c_val, c_val, c_val)
        img_ = cv2.ellipse(img_, center_coor, axesLength,
                           angle, startAngle, endAngle, color, thickness - np.random.randint(2, 5))

        alpha = np.random.randint(40, 80) / 100
        img = cv2.addWeighted(img_, alpha, img, 1 - alpha, 0)
    return img
