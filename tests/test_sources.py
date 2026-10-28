import shutil
import subprocess

import pytest

from ks.data.filter_kitchen import frames_at, title_match
from ks.data.sources.charades import kitchen_rows
from ks.data.sources.epic import select_videos, video_url


def test_epic_picks_one_extension_video_per_participant_closest_to_target():
    info = [
        {"video_id": "P01_01", "duration": "480"},    # EPIC-55 video (2-digit): different host layout, skipped
        {"video_id": "P01_101", "duration": "1200"},
        {"video_id": "P01_102", "duration": "500"},
        {"video_id": "P02_101", "duration": "100"},
        {"video_id": "P02_102", "duration": "600"},   # |600-480| = 120 beats |100-480| = 380
    ]
    got = select_videos(info, per_participant=1, target_seconds=480)
    assert [r["video_id"] for r in got] == ["P01_102", "P02_102"]
    assert got[0]["group"] == "epic:P01"
    assert got[0]["duration"] == 500.0


def test_epic_two_per_participant():
    info = [{"video_id": f"P01_10{i}", "duration": str(d)} for i, d in enumerate([100, 470, 520, 2000])]
    assert [r["video_id"] for r in select_videos(info, per_participant=2, target_seconds=480)] == ["P01_101", "P01_102"]


def test_epic_url_points_at_extension_host():
    assert video_url("P02_113") == "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/P02/videos/P02_113.MP4"


def test_charades_keeps_kitchen_scenes_grouped_by_actor():
    rows = [
        {"id": "AAA01", "subject": "S1", "scene": "Kitchen", "length": "30.5"},
        {"id": "BBB02", "subject": "S1", "scene": "Bedroom", "length": "31.0"},
        {"id": "CCC03", "subject": "S2", "scene": "Kitchen", "length": "12.0"},
    ]
    got = kitchen_rows(rows)
    assert [(r["video_id"], r["group"], r["duration"]) for r in got] == [
        ("charades:AAA01", "charades:S1", 30.5), ("charades:CCC03", "charades:S2", 12.0)]


@pytest.mark.parametrize(
    "title, want",
    [
        ("27 Hilarious Cooking Fail Nominees - FailArmy Hall of Fame (July 2017)", True),
        ("Cut it Out - Knife Fails (October 2018) _ FailArmy", True),
        ("Funny Food and Cooking Fails by FailArmy", True),
        ("Bad Bakers - Kitchen Fails", True),
        ("BBQ Disasters", True),
        ("Backflip Fails (May 2017)", False),       # "bak" must not match "back"
        ("Winter Fails! (January 2017) _ FailArmy", False),
        ("Fired Up! - Fails of the Week", False),     # fire words are not kitchen words
    ],
)
def test_title_match(title, want):
    assert title_match(title) is want


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_frames_at_returns_one_rgb_frame_per_fraction(tmp_path):
    src = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=4:size=320x240:rate=25",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    frames = frames_at(src, (0.25, 0.5, 0.75))
    assert [f.size for f in frames] == [(320, 240)] * 3
    assert all(f.mode == "RGB" for f in frames)
    # testsrc draws a running clock: frames from different times must differ.
    assert frames[0].tobytes() != frames[2].tobytes()
