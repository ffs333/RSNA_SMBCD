import torch
from torch import nn
from torch.nn import functional as F
import monai
import timm

from .nextvit import NextViT, NextViTSupervision, NextViTSupervisionSmall
from .MIP_model import MIPModel


def get_model(_cfg):
    if _cfg.model == 'v0':
        model = ModelV0(backbone=_cfg.backbone, pretrained=_cfg.pretrained, in_channels=_cfg.in_channels,
                        drop_rate=_cfg.drop_rate)
    elif _cfg.model == 'v0_deepsuper':
        model = ModelV0DeepSuper(backbone=_cfg.backbone, pretrained=_cfg.pretrained, in_channels=_cfg.in_channels,
                                 drop_rate=_cfg.drop_rate)
    elif _cfg.model == 'effv2_deepsuper':
        model = EffV2SDeepSuper(backbone=_cfg.backbone, pretrained=_cfg.pretrained, in_channels=_cfg.in_channels,
                                 drop_rate=_cfg.drop_rate)
    elif _cfg.model == 'v1':
        model = ModelV1(backbone=_cfg.backbone, pretrained=_cfg.pretrained, use_meta=_cfg.use_meta,
                        use_act=_cfg.use_act, in_channels=_cfg.in_channels, drop_rate=_cfg.drop_rate)
    elif _cfg.model == 'v2':
        model = ModelV2(backbone=_cfg.backbone, pretrained=_cfg.pretrained, use_meta=_cfg.use_meta,
                        use_act=_cfg.use_act, in_channels=_cfg.in_channels, drop_rate=_cfg.drop_rate)
    elif _cfg.model == 'v3':
        model = ModelV3(backbone=_cfg.backbone, pretrained=_cfg.pretrained, in_channels=_cfg.in_channels,
                        drop_rate=_cfg.drop_rate, use_meta=_cfg.use_meta)

    elif _cfg.model == 'nextvit':
        model = NextViT(stem_chs=[64, 32, 64], depths=[3, 4, 20, 3], path_dropout=0.2, attn_drop=0, drop=0,
                        num_classes=1, strides=[1, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
                        head_dim=32, mix_block_ratio=0.75, use_checkpoint=False)
        ck = torch.load(_cfg.base_path + 'nextvit_wgts/nextvit_base_in1k_224.pth',
                        map_location=torch.device('cpu'))['model']

        model.load_state_dict(dict([(n, p) for n, p in ck.items() if 'proj_head' not in n]), strict=False)

    elif _cfg.model == 'nextvit_deepsuper':
        model = NextViTSupervision(stem_chs=[64, 32, 64], depths=[3, 4, 20, 3], path_dropout=0.2, attn_drop=0, drop=0,
                                   num_classes=1, strides=[1, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
                                   head_dim=32, mix_block_ratio=0.75, use_checkpoint=False)
        ck = torch.load(_cfg.base_path + 'nextvit_wgts/nextvit_base_in1k_224.pth',
                        map_location=torch.device('cpu'))['model']

        model.load_state_dict(dict([(n, p) for n, p in ck.items() if 'proj_head' not in n]), strict=False)

    elif _cfg.model == 'nextvit_s':
        model = NextViT(stem_chs=[64, 32, 64], depths=[3, 4, 10, 3], path_dropout=0.1, attn_drop=0, drop=0,
                        num_classes=1, strides=[1, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
                        head_dim=32, mix_block_ratio=0.75, use_checkpoint=False)
        ck = torch.load(_cfg.base_path + 'nextvit_wgts/nextvit_small_in1k6m_224.pth',
                        map_location=torch.device('cpu'))['model']

        model.load_state_dict(dict([(n, p) for n, p in ck.items() if 'proj_head' not in n]), strict=False)

    elif _cfg.model == 'nextvit_s_deepsuper':
        model = NextViTSupervisionSmall(stem_chs=[64, 32, 64], depths=[3, 4, 10, 3], path_dropout=0.1, attn_drop=0, drop=0,
                                        num_classes=1, strides=[1, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
                                        head_dim=32, mix_block_ratio=0.75, use_checkpoint=False)
        ck = torch.load(_cfg.base_path + 'nextvit_wgts/nextvit_small_in1k6m_224.pth',
                        map_location=torch.device('cpu'))['model']

        model.load_state_dict(dict([(n, p) for n, p in ck.items() if 'proj_head' not in n]), strict=False)

    elif _cfg.model == 'MIP_transformer':
        model = MIPModel(_cfg.backbone, _cfg.mip_ckpt, projection_dim=64, num_classes=1, num_heads=_cfg.mip_heads,
                         feedforward_dim=128, drop_transformer=_cfg.mip_drop, image_shape=(8, 8))

    elif _cfg.model == 'eff_attention':
        model = ATTEfficient(model_arch=_cfg.backbone, drop_rate=_cfg.drop_rate)
        if _cfg.mip_ckpt is not None:
            model.load_state_dict(torch.load(_cfg.mip_ckpt, map_location=torch.device('cpu'))['model'], strict=False)

    elif _cfg.model == 'monai_resnet10':
        model = monai.networks.nets.resnet10(num_classes=1, spatial_dims=2)
    elif _cfg.model == 'monai_resnet18':
        model = monai.networks.nets.resnet18(num_classes=1, spatial_dims=2)

    else:
        raise ValueError('Error in "get_Model" function:',
                         f'Wrong model name. Choose one from ["v1", "v2"]')
    model.eval()
    return model


class ModelV0(nn.Module):
    def __init__(self, backbone: str, pretrained=False, in_channels=3, drop_rate=0.0):
        super().__init__()

        self.drop_rate = drop_rate

        self.encoder = timm.create_model(backbone,
                                         pretrained=pretrained,
                                         in_chans=in_channels,
                                         num_classes=1,
                                         drop_rate=drop_rate
                                         )

    def forward(self, img):
        x = self.encoder(img)
        return x


class EffV2SDeepSuper2(nn.Module):
    def __init__(self, backbone: str, pretrained=False, in_channels=3, drop_rate=0.0):
        super().__init__()

        print(f'Init model EffV2SDeepSuper2')
        NORM_EPS = 1e-3
        self.drop_rate = drop_rate

        self.encoder = timm.create_model(backbone,
                                         pretrained=pretrained,
                                         in_chans=in_channels,
                                         num_classes=1,
                                         drop_rate=drop_rate
                                         )
        self.sup_inds = [2, 3, 4]

        self.sv1 = nn.Sequential(nn.Conv2d(self.encoder.blocks[self.sup_inds[0]][-1].conv_pwl.out_channels, 1280,
                                           kernel_size=(1, 1), stride=(1, 1), bias=False),
                                 nn.BatchNorm2d(1280, eps=NORM_EPS),
                                 nn.SiLU(inplace=True),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten())
        self.sv1_lin = nn.Linear(1280, 1, bias=True)

        self.sv2 = nn.Sequential(nn.Conv2d(self.encoder.blocks[self.sup_inds[1]][-1].conv_pwl.out_channels, 1280,
                                           kernel_size=(1, 1), stride=(1, 1), bias=False),
                                 nn.BatchNorm2d(1280, eps=NORM_EPS),
                                 nn.SiLU(inplace=True),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten())
        self.sv2_lin = nn.Linear(1280, 1, bias=True)

        self.sv3 = nn.Sequential(nn.Conv2d(self.encoder.blocks[self.sup_inds[2]][-1].conv_pwl.out_channels, 1280,
                                           kernel_size=(1, 1), stride=(1, 1), bias=False),
                                 nn.BatchNorm2d(1280, eps=NORM_EPS),
                                 nn.SiLU(inplace=True),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten())
        self.sv3_lin = nn.Linear(1280, 1, bias=True)

    def forward(self, img):
        x = self.encoder.conv_stem(img)
        x = self.encoder.bn1(x)

        features = {}
        for idx in range(len(self.encoder.blocks)):
            x = self.encoder.blocks[idx](x)
            if idx in self.sup_inds:
                features[idx] = x

        sv1 = self.sv1(features[self.sup_inds[0]])
        sv2 = self.sv2(features[self.sup_inds[1]])
        sv3 = self.sv3(features[self.sup_inds[2]])

        x = self.encoder.conv_head(x)
        x = self.encoder.bn2(x)
        x = self.encoder.global_pool(x)
        x = x + sv1 + sv2 + sv3

        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)

        x = self.encoder.classifier(x)

        sv1 = self.sv1_lin(sv1)
        sv2 = self.sv1_lin(sv2)
        sv3 = self.sv1_lin(sv3)

        return x, [sv1, sv2, sv3]


class EffV2SDeepSuper(nn.Module):
    def __init__(self, backbone: str, pretrained=False, in_channels=3, drop_rate=0.0):
        super().__init__()

        print(f'Init model EffV2SDeepSuper')
        NORM_EPS = 1e-3
        self.drop_rate = drop_rate

        self.encoder = timm.create_model(backbone,
                                         pretrained=pretrained,
                                         in_chans=in_channels,
                                         num_classes=1,
                                         drop_rate=drop_rate
                                         )
        self.sup_inds = [2, 3, 4]

        self.sv1 = nn.Sequential(nn.BatchNorm2d(self.encoder.blocks[self.sup_inds[0]][-1].conv_pwl.out_channels, eps=NORM_EPS),
                                 nn.SiLU(inplace=True),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Dropout(p=self.drop_rate),
                                 nn.Linear(self.encoder.blocks[self.sup_inds[0]][-1].conv_pwl.out_channels, 1))

        self.sv2 = nn.Sequential(nn.BatchNorm2d(self.encoder.blocks[self.sup_inds[1]][-1].conv_pwl.out_channels, eps=NORM_EPS),
                                 nn.SiLU(inplace=True),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Dropout(p=self.drop_rate),
                                 nn.Linear(self.encoder.blocks[self.sup_inds[1]][-1].conv_pwl.out_channels, 1))

        self.sv3 = nn.Sequential(nn.BatchNorm2d(self.encoder.blocks[self.sup_inds[2]][-1].conv_pwl.out_channels, eps=NORM_EPS),
                                 nn.SiLU(inplace=True),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Dropout(p=self.drop_rate),
                                 nn.Linear(self.encoder.blocks[self.sup_inds[2]][-1].conv_pwl.out_channels, 1))

    def forward(self, img):
        x = self.encoder.conv_stem(img)
        x = self.encoder.bn1(x)

        features = {}
        for idx in range(len(self.encoder.blocks)):
            x = self.encoder.blocks[idx](x)
            if idx in self.sup_inds:
                features[idx] = x

        x = self.encoder.conv_head(x)
        x = self.encoder.bn2(x)

        x = self.encoder.forward_head(x)
        sv1 = self.sv1(features[self.sup_inds[0]])
        sv2 = self.sv2(features[self.sup_inds[1]])
        sv3 = self.sv3(features[self.sup_inds[2]])

        return x, [sv1, sv2, sv3]


class ModelV0DeepSuper(nn.Module):
    def __init__(self, backbone: str, pretrained=False, in_channels=3, drop_rate=0.0):
        super().__init__()

        NORM_EPS = 1e-3
        self.drop_rate = drop_rate

        self.encoder = timm.create_model(backbone,
                                         pretrained=pretrained,
                                         in_chans=in_channels,
                                         num_classes=1,
                                         drop_rate=drop_rate
                                         )
        self.sup_inds = [3, 4, 5, 6]

        self.sv1 = nn.Sequential(nn.BatchNorm2d(self.encoder.blocks[self.sup_inds[0]][-1].conv_pwl.out_channels,
                                                eps=NORM_EPS),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Dropout(p=drop_rate),
                                 nn.Linear(self.encoder.blocks[self.sup_inds[0]][-1].conv_pwl.out_channels, 1))

        self.sv2 = nn.Sequential(nn.BatchNorm2d(self.encoder.blocks[self.sup_inds[1]][-1].conv_pwl.out_channels,
                                                eps=NORM_EPS),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Dropout(p=drop_rate),
                                 nn.Linear(self.encoder.blocks[self.sup_inds[1]][-1].conv_pwl.out_channels, 1))

        self.sv3 = nn.Sequential(nn.BatchNorm2d(self.encoder.blocks[self.sup_inds[2]][-1].conv_pwl.out_channels,
                                                eps=NORM_EPS),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Dropout(p=drop_rate),
                                 nn.Linear(self.encoder.blocks[self.sup_inds[2]][-1].conv_pwl.out_channels, 1))

        self.sv4 = nn.Sequential(nn.BatchNorm2d(self.encoder.blocks[self.sup_inds[3]][-1].conv_pwl.out_channels,
                                                eps=NORM_EPS),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Dropout(p=drop_rate),
                                 nn.Linear(self.encoder.blocks[self.sup_inds[3]][-1].conv_pwl.out_channels, 1))

    def forward(self, img):
        x = self.encoder.conv_stem(img)
        x = self.encoder.bn1(x)

        features = {}
        for idx in range(len(self.encoder.blocks)):
            x = self.encoder.blocks[idx](x)
            if idx in self.sup_inds:
                features[idx] = x

        x = self.encoder.conv_head(x)
        x = self.encoder.bn2(x)

        x = self.encoder.forward_head(x)
        sv1 = self.sv1(features[self.sup_inds[0]])
        sv2 = self.sv2(features[self.sup_inds[1]])
        sv3 = self.sv3(features[self.sup_inds[2]])
        sv4 = self.sv4(features[self.sup_inds[3]])
        return x, [sv1, sv2, sv3, sv4]


class ModelV1(nn.Module):
    def __init__(self, backbone: str, pretrained=False, use_meta=True,
                 use_act=False, in_channels=3, drop_rate=0.1):
        super().__init__()

        self.drop_rate = drop_rate

        self.encoder = timm.create_model(backbone,
                                         pretrained=pretrained,
                                         in_chans=in_channels,
                                         num_classes=1024)

        if 'res' in backbone:
            pool_features = self.encoder.fc.out_features
        elif 'ffic' in backbone:
            pool_features = self.encoder.classifier.out_features
        else:
            raise ValueError('Dont know last layer name')
        self.use_meta = use_meta
        self.use_act = use_act

        if self.use_act:
            self.act_in = nn.ReLU(inplace=True)
        if self.use_meta:
            self.fc1_ = nn.Linear(pool_features + 5, 1)
        else:
            self.fc1_ = nn.Linear(pool_features, 1)

    def forward(self, img, meta=None):
        x = self.encoder(img)
        if meta is not None:
            x = torch.cat([x, meta], dim=1)
        if self.use_act:
            x = self.act_in(x)

        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc1_(x)
        return x


class ModelV2(nn.Module):
    def __init__(self, backbone: str, pretrained=False, use_meta=True, use_act=False,
                 in_channels=3, drop_rate=0.1):
        super().__init__()

        self.drop_rate = drop_rate
        self.use_act = use_act

        self.encoder = timm.create_model(backbone,
                                         pretrained=pretrained,
                                         in_chans=in_channels,
                                         num_classes=1024)
        pool_features = self.encoder.fc.in_features
        if 'res' in backbone:
            del self.encoder.fc
        elif 'ffic' in backbone:
            del self.encoder.classifier
        else:
            raise ValueError('Dont know what to delete. Last layer name needed')

        self.use_meta = use_meta

        self.act_in = nn.ReLU(inplace=True)
        if self.use_meta:
            self.fc1_ = nn.Linear(pool_features + 5, pool_features // 2)
            self.fc2_ = nn.Linear(pool_features // 2, 1)
        else:
            self.fc1_ = nn.Linear(pool_features, pool_features // 2)
            self.fc2_ = nn.Linear(pool_features, 1)

    def forward(self, img, meta=None):
        x = self.encoder.forward_features(img)
        x = self.encoder.global_pool(x)
        if meta is not None:
            x = torch.cat([x, meta], dim=1)

        x = self.fc1_(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        if self.use_act:
            x = self.act_in(x)
        x = self.fc2_(x)
        return x


class ModelV3(nn.Module):
    def __init__(self, backbone: str, pretrained=False, in_channels=3, drop_rate=0.0, use_meta=False):
        super().__init__()

        self.drop_rate = drop_rate

        self.encoder = timm.create_model(backbone,
                                         pretrained=pretrained,
                                         in_chans=in_channels,
                                         num_classes=1,
                                         drop_rate=drop_rate
                                         )
        if use_meta:
            if 'res' in backbone:
                pool_features = self.encoder.fc.in_features
                del self.encoder.fc
                self.encoder.classifier = nn.Linear(pool_features + 5, 1)
            elif 'ffic' in backbone:
                pool_features = self.encoder.classifier.in_features
                self.encoder.classifier = nn.Linear(pool_features + 5, 1)
            else:
                raise ValueError('Dont know what to delete. Last layer name needed')

    def forward(self, img, meta=None):
        x = self.encoder.forward_features(img)
        x = self.encoder.global_pool(x)
        if meta is not None:
            x = torch.cat([x, meta], dim=1)

        return self.encoder.classifier(x)


class MLPAttentionNetwork(nn.Module):

    def __init__(self, hidden_dim, attention_dim=None):
        super(MLPAttentionNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        if self.attention_dim is None:
            self.attention_dim = self.hidden_dim
        # W * x + b
        self.proj_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=True)
        # v.T
        self.proj_v = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, x):
        H = torch.tanh(self.proj_w(x))
        att_scores = torch.softmax(self.proj_v(H), axis=0)
        attn_x = (x * att_scores).sum(0)

        return attn_x


class ATTEfficient(nn.Module):
    def __init__(self, model_arch, drop_rate, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(model_arch, in_chans=3, pretrained=pretrained, drop_rate=drop_rate)

        cnn_feature = self.encoder.classifier.in_features
        self.encoder.classifier = nn.Identity()

        self.mlp_att = MLPAttentionNetwork(cnn_feature)
        self.logits = nn.Sequential(
            nn.Linear(cnn_feature, 512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 1)
        )

    def forward(self, x, leng):

        features = self.encoder(x)

        attention_values = []
        done = 0
        for idx in range(len(leng)):
            attention_val = self.mlp_att(features[done:done + leng[idx]])
            attention_values.append(attention_val)
            done += leng[idx]
        attention_values = torch.stack(attention_values)

        pred = self.logits(attention_values)
        return pred
