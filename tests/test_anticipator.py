import pytest
import torch
from transformers import VJEPA2Config, VJEPA2Model

from ks.models.vjepa_anticipator import Anticipator, pool_tokens


def tiny_backbone():
    cfg = VJEPA2Config(
        crop_size=32, patch_size=8, tubelet_size=2, frames_per_clip=4, hidden_size=32,
        num_attention_heads=2, num_hidden_layers=1, pred_hidden_size=16, pred_num_attention_heads=2,
        pred_num_hidden_layers=1,
    )
    return VJEPA2Model(cfg)


def test_pool_tokens_averages_2x2_spatial_blocks_per_time_step():
    # grid t=2, h=2, w=4, one channel; token value = its row-major (t, h, w) index.
    tokens = torch.arange(16, dtype=torch.float32).view(1, 16, 1)
    got = pool_tokens(tokens, hw=(2, 4), spatial=2)
    # t0: {0,1,4,5}->2.5  {2,3,6,7}->4.5   t1: {8,9,12,13}->10.5  {10,11,14,15}->12.5
    assert got.flatten().tolist() == [2.5, 4.5, 10.5, 12.5]


def test_pool_tokens_spatial_1_is_identity():
    tokens = torch.randn(2, 16, 3)
    assert torch.equal(pool_tokens(tokens, hw=(2, 4), spatial=1), tokens)


def test_pool_tokens_rejects_grid_that_does_not_divide():
    with pytest.raises(ValueError):
        pool_tokens(torch.randn(1, 12, 3), hw=(3, 4), spatial=2)


def test_anticipator_gives_two_logits_per_clip():
    model = Anticipator(tiny_backbone(), spatial_pool=2, num_heads=2, head_hidden=16).eval()
    clips = torch.randn(3, 4, 3, 32, 32)  # (B, T, C, H, W)
    assert model(clips).shape == (3, 2)


def test_anticipator_encode_returns_pooled_token_grid():
    # 4 frames / tubelet 2 = 2 time steps; 32 px / patch 8 = 4x4 -> pooled 2x2 = 8 tokens.
    model = Anticipator(tiny_backbone(), spatial_pool=2, num_heads=2, head_hidden=16).eval()
    assert model.encode(torch.randn(1, 4, 3, 32, 32)).shape == (1, 8, 32)


def test_head_parameter_names_match_badas_checkpoint():
    # The BADAS-Open checkpoint stores the head under these exact keys; a rename here
    # would make the BADAS-init fine-tune silently start from a random head.
    model = Anticipator(tiny_backbone())
    head_keys = {k for k in model.state_dict() if not k.startswith("backbone.")}
    assert head_keys == {
        "temporal_processor.attention.in_proj_weight", "temporal_processor.attention.in_proj_bias",
        "temporal_processor.attention.out_proj.weight", "temporal_processor.attention.out_proj.bias",
        "temporal_processor.norm.weight", "temporal_processor.norm.bias",
        "classifier.classifier.0.weight", "classifier.classifier.0.bias",
        "classifier.classifier.2.weight", "classifier.classifier.2.bias",
        "classifier.classifier.4.weight", "classifier.classifier.4.bias",
        "classifier.classifier.6.weight", "classifier.classifier.6.bias",
        "classifier.classifier.8.weight", "classifier.classifier.8.bias",
    }


def test_probe_head_weights_load_into_full_model():
    # The frozen probe trains a ProbeHead on cached tokens; its weights must drop into Anticipator unchanged.
    from ks.models.vjepa_anticipator import ProbeHead
    probe = ProbeHead(hidden_size=32, num_heads=2, head_hidden=16)
    model = Anticipator(tiny_backbone(), spatial_pool=2, num_heads=2, head_hidden=16).eval()
    result = model.load_state_dict(probe.state_dict(), strict=False)
    assert result.unexpected_keys == []
    assert all(k.startswith("backbone.") for k in result.missing_keys)
    tokens = torch.randn(2, 8, 32)
    assert torch.allclose(probe.eval()(tokens), model.head(tokens))


def badas_reference_future(predictor, features, n_future):
    """BADAS-Open's _predict_future_features (src/train/video_training.py): one sample at a time,
    zero-padded, context = the present tokens, targets = the next n_future positions."""
    seq, dim = features.shape[1:]
    context, target = torch.arange(seq).unsqueeze(0), torch.arange(seq, seq + n_future).unsqueeze(0)
    out = []
    for i in range(len(features)):
        padded = torch.cat([features[i:i + 1], torch.zeros(1, n_future, dim)], dim=1)
        out.append(predictor(padded, context_mask=[context], target_mask=[target]).last_hidden_state)
    return torch.cat(out)


def test_future_tokens_match_badas_per_sample_predictor_loop():
    torch.manual_seed(0)
    model = Anticipator(tiny_backbone(), num_heads=2, head_hidden=16, future_fraction=0.25).eval()
    features = torch.randn(3, 32, 32)  # 2 time steps x 4 x 4 tokens
    with torch.no_grad():
        got = model.predict_future(features)
        want = badas_reference_future(model.backbone.predictor, features, n_future=8)
    assert got.shape == (3, 8, 32)
    torch.testing.assert_close(got, want)


def test_badas_mode_classifies_present_and_future_tokens_together():
    model = Anticipator(tiny_backbone(), num_heads=2, head_hidden=16, future_fraction=0.25).eval()
    clips = torch.randn(2, 4, 3, 32, 32)
    with torch.no_grad():
        tokens = model.encode(clips)
        want = model.head(torch.cat([tokens, model.predict_future(tokens)], dim=1))
        torch.testing.assert_close(model(clips), want)


def test_future_prediction_needs_the_full_token_grid():
    with pytest.raises(ValueError):
        Anticipator(tiny_backbone(), spatial_pool=2, future_fraction=0.25)
