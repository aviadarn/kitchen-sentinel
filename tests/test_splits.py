import pytest

from ks.data.splits import assign_splits, oops_group


@pytest.mark.parametrize(
    "filename, want",
    [
        ("27 Hilarious Cooking Fail Nominees - FailArmy Hall of Fame (July 2017)20.mp4",
         "27 Hilarious Cooking Fail Nominees - FailArmy Hall of Fame (July 2017)"),
        ("Work Sucks! - Failing At Work (June 2018) _ FailArmy0.mp4", "Work Sucks! - Failing At Work (June 2018) _ FailArmy"),
        ("Work Sucks! - Failing At Work (June 2018) _ FailArmy24.mp4", "Work Sucks! - Failing At Work (June 2018) _ FailArmy"),
        # Title itself ends in a year: "FailArmy 2016" + clip "0". All trailing digits go,
        # so every clip of the compilation still lands in the same group.
        ("Best Funny Kid Fails Compilation Part 2 _ FailArmy 20160.mp4", "Best Funny Kid Fails Compilation Part 2 _ FailArmy"),
        ("Best Funny Kid Fails Compilation Part 2 _ FailArmy 201612.mp4", "Best Funny Kid Fails Compilation Part 2 _ FailArmy"),
        ("videos/oops/Winter Fails! (January 2017) _ FailArmy3.mp4", "Winter Fails! (January 2017) _ FailArmy"),
    ],
)
def test_oops_group_strips_clip_index(filename, want):
    assert oops_group(filename) == want


def test_assign_splits_hits_video_fractions_with_equal_groups():
    sizes = {f"g{i}": 1 for i in range(100)}
    got = assign_splits(sizes, fractions={"train": 0.70, "val": 0.15, "test": 0.15}, seed=0)
    counts = {s: sum(1 for g in got if got[g] == s) for s in ("train", "val", "test")}
    assert counts == {"train": 70, "val": 15, "test": 15}


def test_assign_splits_balances_by_videos_not_groups():
    # One 60-video group + 40 single-video groups. Counting groups would put ~29 groups
    # in train; counting videos must keep train near 70 of 100 videos.
    sizes = {"big": 60, **{f"s{i}": 1 for i in range(40)}}
    got = assign_splits(sizes, fractions={"train": 0.70, "val": 0.15, "test": 0.15}, seed=0)
    videos = {s: sum(sizes[g] for g in got if got[g] == s) for s in ("train", "val", "test")}
    assert got["big"] == "train"
    assert videos == {"train": 70, "val": 15, "test": 15}


def test_assign_splits_is_deterministic_per_seed():
    sizes = {f"g{i}": (i % 5) + 1 for i in range(50)}
    fr = {"train": 0.7, "val": 0.15, "test": 0.15}
    assert assign_splits(sizes, fr, seed=1) == assign_splits(sizes, fr, seed=1)
    assert assign_splits(sizes, fr, seed=1) != assign_splits(sizes, fr, seed=2)


def test_assign_splits_rejects_fractions_not_summing_to_one():
    with pytest.raises(ValueError):
        assign_splits({"g": 1}, {"train": 0.7, "val": 0.2, "test": 0.2}, seed=0)
