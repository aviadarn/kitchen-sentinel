import subprocess

import av
import numpy as np
import pandas as pd

from ks.demo import pick_examples, render
from tests.test_cv import with_youtube_negatives


def preds():
    p = with_youtube_negatives()
    p["score"] = np.where(p.window_label == 1, 0.9, 0.1)
    p.loc[p.video_id == "p0", "score"] = 0.9  # fires from its first window: early
    p.loc[p.video_id.isin(["p1", "p2"]), "score"] = 0.1  # never fires: missed
    p.loc[(p.video_id == "n1") & (p.t_end == 30.0), "score"] = 0.95  # one alarm in normal cooking
    p.loc[(p.video_id == "y0") & (p.t_end == 5.0), "score"] = 0.95  # one alarm on YouTube normal footage
    return p


def test_examples_cover_every_kind_without_repeating_a_clip():
    ex = pick_examples(preds(), dict.fromkeys(range(5), 0.5), per_kind=2, seed=0)
    assert ex.groupby("kind").size().to_dict() == {"on_time": 2, "early": 1, "miss": 2,
                                                   "false_alarm_cooking": 1, "false_alarm_youtube": 1}
    assert ex.video_id.is_unique
    assert set(ex[ex.kind == "miss"].video_id) == {"p1", "p2"}


def test_false_alarm_examples_show_seconds_around_the_alarm_within_the_scored_footage():
    ex = pick_examples(preds(), dict.fromkeys(range(5), 0.5), per_kind=2, seed=0, pad=6.0).set_index("kind")
    cooking, youtube = ex.loc["false_alarm_cooking"], ex.loc["false_alarm_youtube"]
    assert (cooking.start, cooking.stop, cooking.alarm) == (22.0, 36.0, 30.0)  # 2 s window + 6 s each side
    assert (youtube.start, youtube.stop) == (0.0, 10.0)  # clipped to the footage before the failure


def test_render_writes_one_output_frame_per_input_frame(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=8:duration=4",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    times = [2.0, 2.125, 2.25, 3.0, 4.0]
    example = pd.Series(dict(kind="early", video_id="clip", path=str(clip), start=0.0, stop=4.0, fold=0,
                             title="test clip", alarm=np.nan))
    curve = pd.DataFrame(dict(t_end=times, score=[0.1, 0.2, 0.9, 0.95, 0.3]))
    out = render(example, curve, tau=0.5, accidents=[(2.5, 3.0)], out_path=tmp_path / "out.mp4")
    with av.open(str(out)) as c:
        assert sum(1 for _ in c.decode(video=0)) == 32
