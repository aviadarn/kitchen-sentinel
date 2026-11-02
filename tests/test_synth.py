import pandas as pd
import pytest

from ks.synth.wan_pairs import fit_size, load_jobs


def test_fit_size_keeps_the_aspect_near_the_target_area_on_a_32_pixel_grid():
    w, h = fit_size(640, 480, area=1280 * 704)
    assert w % 32 == 0 and h % 32 == 0
    assert w / h == pytest.approx(4 / 3, rel=0.03)
    assert abs(w * h - 1280 * 704) / (1280 * 704) < 0.06


def test_fit_size_handles_portrait_sources():
    w, h = fit_size(480, 854, area=1280 * 704)
    assert h > w and w % 32 == 0 and h % 32 == 0


def test_jobs_need_both_prompts_and_a_known_hazard(tmp_path):
    path = tmp_path / "jobs.csv"
    pd.DataFrame([dict(pair="p01", video_id="charades:AB12C", t=3.0, hazard="fire",
                       accident="the pan bursts into flames", safe="he keeps stirring")]).to_csv(path, index=False)
    jobs = load_jobs(path)
    assert jobs.iloc[0].pair == "p01"
    pd.DataFrame([dict(pair="p02", video_id="charades:AB12C", t=3.0, hazard="explosion",
                       accident="x", safe="y")]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="hazard"):
        load_jobs(path)


def test_gallery_shows_each_start_frame_beside_both_continuations(tmp_path):
    from ks.synth.wan_pairs import write_gallery

    jobs = pd.DataFrame([dict(pair="p01", video_id="charades:AB12C", t=3.0, hazard="fire",
                              accident="the pan bursts into flames", safe="he keeps stirring")])
    page = write_gallery(jobs, tmp_path)
    html = page.read_text()
    for name in ("p01_source.jpg", "p01_safe.mp4", "p01_accident.mp4", "the pan bursts into flames"):
        assert name in html


def test_held_out_auc_is_high_for_separable_features_and_near_chance_for_noise():
    import numpy as np

    from ks.synth.detect import held_out_auc

    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 8)
    y = np.tile([0, 1], 40)
    separable = rng.normal(0, 1, (80, 5)) + y[:, None] * 3.0
    noise = rng.normal(0, 1, (80, 5))
    assert held_out_auc(separable, y, groups) > 0.95
    assert 0.3 < held_out_auc(noise, y, groups) < 0.7


def test_save_video_writes_float_frames_as_h264_at_the_given_rate(tmp_path):
    import av
    import numpy as np

    from ks.synth.wan_pairs import save_video

    frames = np.random.default_rng(0).random((12, 64, 96, 3)).astype(np.float32)  # diffusers' np output
    save_video(frames, tmp_path / "out.mp4", fps=24)
    with av.open(str(tmp_path / "out.mp4")) as c:
        stream = c.streams.video[0]
        rate = stream.average_rate
        assert (stream.width, stream.height, rate.numerator / rate.denominator) == (96, 64, 24)
        assert sum(1 for _ in c.decode(video=0)) == 12
