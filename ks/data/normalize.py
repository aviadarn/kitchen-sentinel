"""Re-encode any source video to the model's input format: 8 fps, 256x256, no audio.

Every source is squash-resized the same way. Cropping would cut the top and bottom off vertical
phone clips; letterboxing would add black bars that only YouTube positives have, a free shortcut.
"""

import subprocess


def normalize_video(src, dst, fps=8, size=256):
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-an",
         "-vf", f"fps={fps},scale={size}:{size},setsar=1",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(dst)],
        check=True,
    )
