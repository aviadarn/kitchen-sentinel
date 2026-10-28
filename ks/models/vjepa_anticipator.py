"""V-JEPA2 backbone + BADAS-Open's released head.

BADAS-Open's model card describes "12 learned queries", but the checkpoint loader in their code
(badas/utils/video.py, EnhancedVideoClassifier) builds self-attention over all patch tokens, a
LayerNorm, mean pooling and a 3-layer MLP to 2 logits. We mirror the code, with the same parameter
names, so a BADAS-initialised run can load their full checkpoint.
"""

import torch.nn.functional as F
from torch import nn


def pool_tokens(tokens, hw, spatial):
    """Average-pool patch tokens over `spatial` x `spatial` blocks within each time step.

    V-JEPA2 emits tokens in row-major (t, h, w) order. Pooling 2x2 cuts 2048 tokens to 512, which
    keeps cached features for the frozen probe at a manageable size.
    """
    if spatial == 1:
        return tokens
    b, n, d = tokens.shape
    h, w = hw
    if n % (h * w) or h % spatial or w % spatial:
        raise ValueError(f"{n} tokens do not form a (t, {h}, {w}) grid poolable by {spatial}")
    grid = tokens.reshape(b, n // (h * w), h, w, d).permute(0, 4, 1, 2, 3)
    pooled = F.avg_pool3d(grid, kernel_size=(1, spatial, spatial))
    return pooled.flatten(2).transpose(1, 2)


class TemporalAttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, tokens):
        out, _ = self.attention(tokens, tokens, tokens)
        return self.norm(out).mean(dim=1)


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        layers, dim = [], in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout)]
            dim = hidden_dim
        layers.append(nn.Linear(dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


class Anticipator(nn.Module):
    """clips (B, T, C, H, W) -> 2 logits (safe, danger).

    `encode` and `head` are split so the frozen-probe stage can cache `encode` outputs once.
    """

    def __init__(self, backbone, spatial_pool=1, num_heads=8, head_hidden=768, head_layers=3, num_classes=2,
                 dropout=0.1):
        super().__init__()
        cfg = backbone.config
        self.backbone = backbone
        self.spatial_pool = spatial_pool
        self.hw = (cfg.crop_size // cfg.patch_size, cfg.crop_size // cfg.patch_size)
        self.temporal_processor = TemporalAttentionPooling(cfg.hidden_size, num_heads)
        self.classifier = MLPHead(cfg.hidden_size, head_hidden, head_layers, num_classes, dropout)

    def encode(self, clips):
        tokens = self.backbone(pixel_values_videos=clips, skip_predictor=True).last_hidden_state
        return pool_tokens(tokens, self.hw, self.spatial_pool)

    def head(self, tokens):
        return self.classifier(self.temporal_processor(tokens))

    def forward(self, clips):
        return self.head(self.encode(clips))
