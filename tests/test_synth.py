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
