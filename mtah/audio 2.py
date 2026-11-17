"""Read a recording's sound through one pinned ffmpeg, in chunks, as 48 kHz stereo floats.

ffmpeg is found in this order: the MTAH_FFMPEG environment variable, the static build that the
imageio-ffmpeg package carries (pinned, the same on every computer of one platform), then ffmpeg on
the PATH. Every caller uses the same decoding settings, so the sound reaches the detector the same
way each time.
"""

import os
import shutil
import subprocess
from typing import Iterator, Optional

import numpy as np

SR = 48000
CHANNELS = 2
PROJECT_TOOLS = os.path.join(os.path.dirname(__file__), "..", "..", ".audio-tools", "lib")


def find_ffmpeg() -> str:
    if os.environ.get("MTAH_FFMPEG"):
        return os.environ["MTAH_FFMPEG"]
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    for root, _, files in os.walk(os.path.abspath(PROJECT_TOOLS)):
        for name in files:
            if name.startswith("ffmpeg-") and os.access(os.path.join(root, name), os.X_OK):
                return os.path.join(root, name)
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError("ffmpeg was not found. Install it with: python3 -m pip install imageio-ffmpeg==0.8.1")


def chunks(path: str, chunk_s: float = 60.0, ffmpeg: Optional[str] = None) -> Iterator[np.ndarray]:
    """[n, 2] float32 pieces of the first audio stream. Raises RuntimeError if there is none."""
    command = [ffmpeg or find_ffmpeg(), "-nostdin", "-v", "error", "-i", path, "-vn", "-map", "0:a:0",
               "-ac", str(CHANNELS), "-ar", str(SR), "-f", "s16le", "-acodec", "pcm_s16le", "-"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    size = int(chunk_s * SR) * CHANNELS * 2
    got_any = False
    try:
        while True:
            raw = process.stdout.read(size)
            if not raw:
                break
            got_any = True
            samples = np.frombuffer(raw[: len(raw) - len(raw) % 4], dtype="<i2")
            yield samples.reshape(-1, CHANNELS).astype(np.float32) / 32768.0
    finally:
        process.stdout.close()
        error = process.stderr.read().decode(errors="replace").strip()
        process.wait()
    if process.returncode != 0 or not got_any:
        raise RuntimeError(f"no sound could be read from {os.path.basename(path)}: {error or 'no audio stream'}")


def to_flac(source: str, target: str, ffmpeg: Optional[str] = None) -> None:
    """Keep only the sound of a video, losslessly compressed, at the detector's settings."""
    command = [ffmpeg or find_ffmpeg(), "-nostdin", "-v", "error", "-n", "-i", source, "-vn", "-map", "0:a:0",
               "-ac", str(CHANNELS), "-ar", str(SR), "-sample_fmt", "s16", "-c:a", "flac", target]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"could not take the sound out of {os.path.basename(source)}: {result.stderr.strip()}")


def describe(path: str, ffmpeg: Optional[str] = None) -> dict:
    """What a file holds, read from ffmpeg's own summary: whether it has pictures and sound, and the
    first audio stream's sample rate and channel layout (as words, e.g. "stereo")."""
    result = subprocess.run([ffmpeg or find_ffmpeg(), "-nostdin", "-hide_banner", "-i", path],
                            capture_output=True, text=True)
    text = result.stderr
    out = {"video": " Video: " in text, "audio": " Audio: " in text, "rate": None, "layout": None}
    for line in text.splitlines():
        if " Audio: " in line:
            parts = [p.strip() for p in line.split(" Audio: ", 1)[1].split(",")]
            for i, part in enumerate(parts):
                if part.endswith(" Hz"):
                    out["rate"] = int(part.split()[0])
                    out["layout"] = parts[i + 1] if i + 1 < len(parts) else None
            break
    return out


def first_times(path: str, ffmpeg: Optional[str] = None) -> dict:
    """When the first picture and the first sound arrive, in seconds on the file's own clock (None if
    there is none). video - audio is where video frame 0 sits on the sound's clock."""
    out = {}
    for kind, stream, flt, count in (("video", "0:v:0", "showinfo", "-frames:v"), ("audio", "0:a:0", "ashowinfo", "-frames:a")):
        command = [ffmpeg or find_ffmpeg(), "-nostdin", "-hide_banner", "-i", path, "-map", stream, count, "1",
                   "-vf" if kind == "video" else "-af", flt, "-f", "null", "-"]
        text = subprocess.run(command, capture_output=True, text=True).stderr
        found = [line.split("pts_time:", 1)[1].split()[0] for line in text.splitlines() if "pts_time:" in line]
        out[kind] = float(found[0]) if found else None
    return out
