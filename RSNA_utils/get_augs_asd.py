import torch
import monai.transforms as T
import torchvision


def get_transforms(_cfg):

    transforms_0 = T.Compose([
        # Main
        T.Lambda(func=lambda x: x / 255.),
        # T.NormalizeIntensity(subtrahend=125, divisor=125),

        T.RandFlip(prob=_cfg.flip_prob_0, spatial_axis=1),

        T.RandZoom(prob=_cfg.zoom_prob_0, min_zoom=0.85, max_zoom=1.25),
        T.RandRotate(prob=_cfg.rotate_prob_0, range_x=0.4),

        # T.OneOf([
        #        T.RandGaussianSmooth(sigma_x=(1., 2.), sigma_y=(1., 2.), prob=_cfg.gaus_prob_0, approx='erf'),
        #        T.RandGibbsNoise(prob=_cfg.gibbs_prob_0, alpha=(0.5, 0.8))]),

        T.OneOf([
                T.Rand2DElastic(spacing=(_cfg.img_size//8, _cfg.img_size//8),
                                prob=_cfg.elastic_prob_0, magnitude_range=(1, 2), padding_mode='zeros'),
                T.RandAffine(prob=_cfg.affine_prob_0, rotate_range=None, shear_range=(0.05, 0.2))]),

        T.RandAdjustContrast(prob=_cfg.contr_prob_0, gamma=(0.9, 1.1)),
        #T.RandCoarseDropout(holes=2, spatial_size=_cfg.img_size//80, max_spatial_size=_cfg.img_size//60,
        #                    fill_value=None, max_holes=50, prob=_cfg.drop_prob_0),

        T.EnsureType(dtype=torch.float32)
    ])

    transforms_1 = T.Compose([
        # Main
        T.Lambda(func=lambda x: x / 255.),
        # T.NormalizeIntensity(subtrahend=125, divisor=125),
        T.RandFlip(prob=_cfg.flip_prob_0, spatial_axis=1),

        T.RandZoom(prob=_cfg.zoom_prob_1, min_zoom=0.85, max_zoom=1.25),
        T.RandRotate(prob=_cfg.rotate_prob_1, range_x=0.4),

        # T.OneOf([
        #        T.RandGaussianSmooth(sigma_x=(1., 2.), sigma_y=(1., 2.), prob=_cfg.gaus_prob_1, approx='erf'),
        #        T.RandGibbsNoise(prob=_cfg.gibbs_prob_1, alpha=(0.5, 0.8))]),

        T.OneOf([
                T.Rand2DElastic(spacing=(_cfg.img_size//8, _cfg.img_size//8),
                                prob=_cfg.elastic_prob_1, magnitude_range=(1, 2), padding_mode='zeros'),
                T.RandAffine(prob=_cfg.affine_prob_1, rotate_range=None, shear_range=(0.05, 0.2))]),

        T.RandAdjustContrast(prob=_cfg.contr_prob_1, gamma=(0.9, 1.1)),
        #T.RandCoarseDropout(holes=2, spatial_size=_cfg.img_size//80, max_spatial_size=_cfg.img_size//70,
        #                    fill_value=None, max_holes=50, prob=_cfg.drop_prob_1),

        T.EnsureType(dtype=torch.float32)
    ])

    valid_transforms = T.Compose([
        # Main
        T.Lambda(func=lambda x: x / 255.),
        # T.NormalizeIntensity(subtrahend=125, divisor=125),
        T.EnsureType(dtype=torch.float32)
    ])
    return transforms_0, transforms_1, valid_transforms
