"""Group-level train/val/test splits, so correlated videos never straddle two splits."""

import random
import re
from pathlib import PurePosixPath


def oops_group(filename):
    """Compilation title of an Oops! clip, e.g. '... (July 2017)20.mp4' -> '... (July 2017)'.

    Clips are named <compilation title><clip index>. Some titles end in digits themselves
    ('FailArmy 2016' + clip '0' = 'FailArmy 20160'), so all trailing digits are stripped. That can
    merge two compilations into one group, but never splits one compilation across groups.
    """
    stem = PurePosixPath(filename).stem
    return re.sub(r"\d+$", "", stem).rstrip()


def assign_splits(group_sizes, fractions, seed):
    """Map each group to a split so that each split's share of *videos* tracks `fractions`.

    Largest groups are placed first (ties in random order from `seed`), each into the split
    furthest below its target video count.
    """
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    groups = list(group_sizes)
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda g: -group_sizes[g])  # stable sort keeps the shuffled order among ties
    total = sum(group_sizes.values())
    target = {s: f * total for s, f in fractions.items()}
    filled = dict.fromkeys(fractions, 0)
    assignment = {}
    for g in groups:
        split = max(fractions, key=lambda s: target[s] - filled[s])
        assignment[g] = split
        filled[split] += group_sizes[g]
    return assignment
