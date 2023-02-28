import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss


def get_loss_func(cfg):
    if cfg.loss == 'bce':
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_wgt))
    elif cfg.loss == 'diff_bce':
        loss_fn = DiffBCELoss(pos_wgt=cfg.pos_wgt, dif_wgt=cfg.dif_wgt)
    elif cfg.loss == 'LMF':
        loss_fn = MyLMFLoss(cfg)
    elif cfg.loss == 'LMF_EXT':
        loss_fn = MyLMFLossEXT(cfg)
    elif cfg.loss == 'LMFBCE':
        loss_fn = MyLMFBCELoss(cfg)

    else:
        raise ValueError('Error in "get_loss_func" function:',
                         f'Wrong loss name. Choose one from ["bce", "diff_bce", "LMF", "LMF_EXT"] ')

    return loss_fn


class DiffBCELoss(nn.Module):
    def __init__(self, pos_wgt=10, dif_wgt=0.5):
        super().__init__()

        self.neg_w = dif_wgt
        self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(pos_wgt))

    def forward(self, y_pred, y_true, diff):
        loss = self.bce(y_pred, y_true)
        loss *= torch.where(diff > 0.5, self.neg_w, 1.)

        return loss.mean()


class FocalLoss(torch.nn.Module):
    def __init__(self, device, alpha=0.25, gamma=2.0, weight=1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = torch.tensor(weight, device=device)

    def forward(self, logits, targets):
        # compute the binary cross-entropy loss
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=self.weight)

        # compute the focal loss
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        # return the mean focal loss
        return focal_loss.mean()


class MyLDAMLoss(nn.Module):

    def __init__(self, device, cls_num_list, max_m=0.5, weight=None, s=30):
        super(MyLDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.tensor(m_list, device=device).float()
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = torch.tensor(weight, device=device)

    def forward(self, x, target):
        batch_m = torch.where(target == 1, self.m_list[1], self.m_list[0]).view(-1, 1)
        x_m = x - batch_m

        return F.binary_cross_entropy_with_logits(x_m * self.s, target, weight=self.weight)


class MyLMFLoss(nn.Module):

    def __init__(self, cfg):
        super(MyLMFLoss, self).__init__()
        self.ldam = MyLDAMLoss(device=cfg.device, cls_num_list=[cfg.neg_samples, cfg.pos_samples], max_m=cfg.ldam_max_m,
                               weight=cfg.pos_wgt, s=cfg.ldam_s)

        self.focal = FocalLoss(device=cfg.device, alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, weight=cfg.pos_wgt)

        self.w_ldam = cfg.w_ldam
        self.w_focal = cfg.w_focal

    def forward(self, x, target):
        ldam_val = self.ldam(x, target)
        focal_val = self.focal(x, target)

        return ldam_val * self. w_ldam + focal_val * self.w_focal


class MyLMFBCELoss(nn.Module):

    def __init__(self, cfg):
        super(MyLMFBCELoss, self).__init__()
        self.ldam = MyLDAMLoss(device=cfg.device, cls_num_list=[cfg.neg_samples, cfg.pos_samples], max_m=cfg.ldam_max_m,
                               weight=cfg.pos_wgt, s=cfg.ldam_s)

        self.focal = FocalLoss(device=cfg.device, alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, weight=cfg.pos_wgt)

        self.bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(cfg.pos_wgt))

        self.w_ldam = cfg.w_ldam
        self.w_focal = cfg.w_focal
        self.w_bce = cfg.w_bce

    def forward(self, x, target):
        ldam_val = self.ldam(x, target)
        focal_val = self.focal(x, target)
        bce_val = self.bce(x, target)

        return ldam_val * self. w_ldam + focal_val * self.w_focal + bce_val * self.w_bce


class MyLMFandBCELoss(nn.Module):

    def __init__(self, cfg):
        super(MyLMFandBCELoss, self).__init__()
        self.ldam = MyLDAMLoss(device=cfg.device, cls_num_list=[cfg.neg_samples, cfg.pos_samples], max_m=cfg.ldam_max_m,
                               weight=cfg.pos_wgt, s=cfg.ldam_s)

        self.focal = FocalLoss(device=cfg.device, alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, weight=cfg.pos_wgt)

        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_wgt))

        self.w_ldam = cfg.w_ldam
        self.w_focal = cfg.w_focal

        cycles = cfg.epochs  # how many sine cycles
        total_steps = cfg.num_iters  # how many datapoints to generate

        length = np.pi * 2 * cycles
        wave = np.sin(np.arange(0, length, length / total_steps))
        self.wave = (wave + 2) / 4
        self.global_step = 0

    def forward(self, x, target):
        ldam_val = self.ldam(x, target)
        focal_val = self.focal(x, target)
        bce_val = self.bce(x, target)

        lmf_val = (ldam_val * self.w_ldam + focal_val * self.w_focal) / 2

        k = self.wave[self.global_step]
        # print(k)
        return lmf_val * k + bce_val * (1 - k)


class MyLMFLossEXT(nn.Module):

    def __init__(self, cfg):
        super(MyLMFLossEXT, self).__init__()
        self.ldam = MyLDAMLoss(device=cfg.device, cls_num_list=[cfg.neg_samples, cfg.pos_samples], max_m=cfg.ldam_max_m,
                               weight=cfg.pos_wgt, s=cfg.ldam_s, reduction='none')

        self.focal = FocalLoss(device=cfg.device, alpha=cfg.focal_alpha, gamma=cfg.focal_gamma,
                               weight=cfg.pos_wgt, reduction='none')

        self.w_ldam = cfg.w_ldam
        self.w_focal = cfg.w_focal
        self.exp_w = 1/cfg.pos_wgt

    def forward(self, x, target, exp):
        ldam_val = self.ldam(x, target)
        focal_val = self.focal(x, target)
        sum_loss = ldam_val * self.w_ldam + focal_val * self.w_focal

        exp = torch.where(exp == 1., self.exp_w, 1.)
        sum_loss *= exp
        return sum_loss.mean()
