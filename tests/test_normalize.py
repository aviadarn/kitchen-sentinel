import shutil
import subprocess

import av
import pytest

from ks.data.normalize import normalize_video

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def make_video(path, seconds, fps, size):
    w, h = size
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size={w}x{h}:rate={fps}",
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


def probe(path):
    with av.open(str(path)) as c:
        frames = [f for f in c.decode(video=0)]
        s = c.streams.video[0]
        return len(frames), frames[0].width, frames[0].height, float(s.average_rate)


def test_normalize_resamples_to_8fps_and_squashes_to_256(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(src, seconds=5, fps=30, size=(640, 360))  # 16:9 source
    normalize_video(src, dst, fps=8, size=256)
    n, w, h, rate = probe(dst)
    assert (w, h) == (256, 256)  # squashed, not cropped or letterboxed
    assert rate == 8.0
    assert n == 40


def test_normalize_handles_vertical_phone_video(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(src, seconds=2, fps=24, size=(360, 640))
    normalize_video(src, dst, fps=8, size=256)
    n, w, h, _ = probe(dst)
    assert (w, h, n) == (256, 256, 16)
