"""V-JEPA2 backbone + BADAS-Open's released head.

BADAS-Open's model card describes "12 learned queries", but the checkpoint loader in their code
(badas/utils/video.py, EnhancedVideoClassifier) builds self-attention over all patch tokens, a
LayerNorm, mean pooling and a 3-layer MLP to 2 logits. We mirror the code, with the same parameter
names, so a BADAS-initialised run can load their full checkpoint.
"""

import torch
import torch.nn.functional as F
from torch import nn

BACKBONE = "facebook/vjepa2-vitl-fpc16-256-ssv2"


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


class ProbeHead(nn.Module):
    """The Anticipator's head alone, for the frozen probe trained on cached tokens. Same parameter
    names, so `Anticipator.load_state_dict(probe.state_dict(), strict=False)` installs it."""

    def __init__(self, hidden_size=1024, num_heads=8, head_hidden=768, head_layers=3, num_classes=2, dropout=0.1):
        super().__init__()
        self.temporal_processor = TemporalAttentionPooling(hidden_size, num_heads)
        self.classifier = MLPHead(hidden_size, head_hidden, head_layers, num_classes, dropout)

    def forward(self, tokens):
        return self.classifier(self.temporal_processor(tokens))


class Anticipator(nn.Module):
    """clips (B, T, C, H, W) -> 2 logits (safe, danger).

    `encode` and `head` are split so the frozen-probe stage can cache `encode` outputs once.

    With `future_fraction`, the head also sees tokens that V-JEPA2's predictor imagines for the next
    `future_fraction` x (clip length), as BADAS-Open does (use_future_prediction, 'concat': 1 s ahead
    of a clip it treats as 4 s, so 512 future tokens after 2048 present ones).
    """

    def __init__(self, backbone, spatial_pool=1, num_heads=8, head_hidden=768, head_layers=3, num_classes=2,
                 dropout=0.1, future_fraction=None):
        super().__init__()
        cfg = backbone.config
        if future_fraction and spatial_pool != 1:
            raise ValueError("the predictor needs the full token grid; use spatial_pool=1 with future_fraction")
        self.backbone = backbone
        self.spatial_pool = spatial_pool
        self.future_fraction = future_fraction
        self.hw = (cfg.crop_size // cfg.patch_size, cfg.crop_size // cfg.patch_size)
        self.temporal_processor = TemporalAttentionPooling(cfg.hidden_size, num_heads)
        self.classifier = MLPHead(cfg.hidden_size, head_hidden, head_layers, num_classes, dropout)

    def encode(self, clips):
        tokens = self.backbone(pixel_values_videos=clips, skip_predictor=True).last_hidden_state
        return pool_tokens(tokens, self.hw, self.spatial_pool)

    def predict_future(self, tokens):
        """Predictor output for the `future_fraction` x len(tokens) positions after the present ones."""
        b, seq, dim = tokens.shape
        n_future = max(int(seq * self.future_fraction), 1)
        context = torch.arange(seq, device=tokens.device).expand(b, -1)
        target = torch.arange(seq, seq + n_future, device=tokens.device).expand(b, -1)
        padded = torch.cat([tokens, tokens.new_zeros(b, n_future, dim)], dim=1)
        return self.backbone.predictor(padded, context_mask=[context], target_mask=[target]).last_hidden_state

    def head(self, tokens):
        return self.classifier(self.temporal_processor(tokens))

    def forward(self, clips):
        tokens = self.encode(clips)
        if self.future_fraction:
            tokens = torch.cat([tokens, self.predict_future(tokens)], dim=1)
        return self.head(tokens)


def load_badas(backbone):
    """Anticipator holding BADAS-Open's fine-tuned backbone and head (gated: accept the terms at
    https://huggingface.co/nexar-ai/BADAS-Open). Its top-level `predictor.*` tensors are an alias of
    `backbone.predictor.*` (same module registered twice), so they are left out."""
    from huggingface_hub import hf_hub_download

    ckpt = torch.load(hf_hub_download("nexar-ai/BADAS-Open", "weights/badas_open.pth"), map_location="cpu",
                      weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt.get("state_dict") or ckpt
    cfg = ckpt["config"] if isinstance(ckpt["config"], dict) else vars(ckpt["config"])
    future = None
    if cfg.get("use_future_prediction"):
        if cfg.get("predictor_combination_method", "concat") != "concat":
            raise NotImplementedError(f"BADAS combination {cfg['predictor_combination_method']!r}")
        future = cfg.get("future_prediction_seconds", 1.0) / (cfg["frame_count"] / cfg.get("original_fps", 4))
    model = Anticipator(backbone, spatial_pool=1, future_fraction=future)
    result = model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()}, strict=False)
    if result.missing_keys:
        raise RuntimeError(f"BADAS checkpoint lacks {len(result.missing_keys)} tensors, e.g. {result.missing_keys[:3]}")
    return model
