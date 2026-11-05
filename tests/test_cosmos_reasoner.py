import torch

from ks.models.cosmos_reasoner import answer_probability, subsample


def test_subsample_keeps_every_other_frame_ending_on_the_last():
    # 16 frames at 8 fps -> 8 at the model card's 4 fps; the last frame is the moment being scored
    windows = torch.arange(16).view(1, 16, 1, 1, 1).expand(2, 16, 2, 2, 3)
    out = subsample(windows, src_fps=8.0, fps=4.0)
    assert out.shape == (2, 8, 2, 2, 3)
    assert out[0, :, 0, 0, 0].tolist() == [1, 3, 5, 7, 9, 11, 13, 15]


def test_answer_probability_compares_yes_with_no_tokens_only():
    logits = torch.full((3, 6), -10.0)
    logits[0, [0, 1]] = 5.0  # yes variants dominate
    logits[1, [2, 3]] = 5.0  # no variants dominate
    logits[2, [0, 2]] = 5.0  # one of each: undecided
    p_yes, mass = answer_probability(logits, yes_ids=[0, 1], no_ids=[2, 3])
    assert p_yes[0] > 0.99 and p_yes[1] < 0.01
    assert abs(p_yes[2].item() - 0.5) < 1e-6
    assert (mass > 0.99).all()


def test_answer_probability_reports_mass_on_other_tokens():
    # The model wanted to say something else (token 5): P(yes) stays defined, the low mass flags it
    logits = torch.zeros(1, 6)
    logits[0, 5] = 10.0
    p_yes, mass = answer_probability(logits, yes_ids=[0], no_ids=[1])
    assert abs(p_yes.item() - 0.5) < 1e-6
    assert mass.item() < 0.01
