import torch
import timm
import torch.nn as nn


class MIPModel(nn.Module):
    def __init__(
        self,
        backbone,
        checkpoint,
        projection_dim=64,
        num_classes=1,
        num_heads=2,
        feedforward_dim=128,
        drop_transformer=0.25,
        image_shape=(8, 8),
    ):
        super().__init__()

        self.image_shape = image_shape
        self.encoder = timm.create_model(backbone,
                                         pretrained=False,
                                         in_chans=3,
                                         num_classes=1,
                                         drop_rate=0.
                                         )
        ck = torch.load(checkpoint, map_location=torch.device('cpu'))['model']
        self.encoder.load_state_dict(dict([(n.split('encoder.')[-1], p) for n, p in ck.items()]), strict=False)

        feature_dim = self.encoder.conv_head.out_channels
        self.norm = nn.BatchNorm2d(feature_dim)
        # self.projection = nn.Conv2d(feature_dim, projection_dim, (1, 1))

        transformer_dim = projection_dim
        self.pos_encoding = nn.Linear(feature_dim, transformer_dim)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            dim_feedforward=feedforward_dim,
            nhead=num_heads,
            dropout=drop_transformer,
        )
        self.classifier = nn.Linear(feature_dim + projection_dim, num_classes)

    def _apply_transformer(self, image_feats: torch.Tensor, lens):
        att_emb = self.pos_encoding(image_feats)
        att_emb = torch.split(att_emb, split_size_or_sections=lens, dim=0)
        att_emb, mask = self.pad_tensor(att_emb, lens)
        return self.transformer(att_emb)

    @staticmethod
    def pad_tensor(t, length):
        batch_size = len(t)
        dim = t[0].shape[-1]
        max_L = max(length)

        pad_t = torch.zeros((batch_size, max_L, dim)).to(t[0].device)
        pad_mask = torch.zeros((batch_size, max_L)).to(t[0].device)
        for b in range(batch_size):
            pad_t[b, :length[b]] = t[b]
            pad_mask[b, :length[b]] = 0
        pad_mask = pad_mask > 0.5
        return pad_t, pad_mask

    @staticmethod
    def unpad_tensor(pad_t, length):
        batch_size = len(pad_t)
        t = []
        for b in range(batch_size):
            t.append(pad_t[b, :length[b]])
        return t

    def forward(self, images, lens):
        B = len(lens)   # Batch size

        image_feats = self.encoder.forward_features(images)
        image_feats_proj = self.encoder.global_pool(image_feats)  # [N, L]

        # Apply transformer
        image_feats_trans = self._apply_transformer(image_feats_proj, lens)  # [B, C, L]

        image_feats_trans = self.unpad_tensor(image_feats_trans, lens)  # [B, C, L]
        image_feats_proj = torch.split(image_feats_proj, split_size_or_sections=lens, dim=0)   # [B, C, L]

        pool = []
        for b in range(B):
            p = torch.cat([image_feats_proj[b], image_feats_trans[b]], -1).sum(0)
            pool.append(p)
        pool = torch.stack(pool)  # [B, L]

        return self.classifier(pool)

