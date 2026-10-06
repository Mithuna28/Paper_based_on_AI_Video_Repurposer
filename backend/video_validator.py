import sys
import json
import re
import subprocess
from pathlib import Path


def _black_pixel_percentage(file_path: Path, timestamp: float) -> int:
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "info",
            "-ss", f"{timestamp:.3f}",
            "-i", str(file_path),
            "-vf", "crop=iw:ih*0.55:0:ih*0.225,blackframe=amount=0:threshold=80",
            "-frames:v", "1", "-an", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    matches = re.findall(r"pblack:(\d+)", result.stderr)
    if result.returncode != 0 or not matches:
        raise ValueError(
            f"Could not inspect video frame at {timestamp:.2f}s: {result.stderr.strip()}"
        )
    return int(matches[0])


def validate_video_has_visible_content(file_path: Path, duration: float = None) -> None:
    file_path = Path(file_path).resolve()
    if duration is None:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(file_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        duration = float(result.stdout.strip())

    sample_times = [duration * percentage for percentage in (0.1, 0.35, 0.6, 0.85)]
    black_percentages = [
        _black_pixel_percentage(file_path, timestamp) for timestamp in sample_times
    ]
    visible_samples = sum(percentage < 99 for percentage in black_percentages)
    print(f"Black-pixel percentages at sampled frames: {black_percentages}")
    if visible_samples < 2:
        raise ValueError(
            "Video contains no reliably visible picture: at least 99% of sampled "
            "pixels are near-black in most frames."
        )

def validate_video_file(file_path: Path) -> bool:
    """
    Comprehensive pipeline video validation function.
    Verifies stream existence, resolution, codec, pixel format, duration,
    and checks that video frames are not black.
    """
    file_path = Path(file_path).resolve()
    print(f"\n--- VALIDATING VIDEO: {file_path.name} ---")

    if not file_path.exists():
        raise ValueError(f"Validation Failed [Stage: Output Check]: File {file_path} does not exist.")

    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(file_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise ValueError(f"Validation Failed [Stage: FFprobe]: Could not probe video file: {res.stderr}")

    try:
        data = json.loads(res.stdout)
    except Exception as e:
        raise ValueError(f"Validation Failed [Stage: JSON Parse]: Invalid ffprobe output: {e}")

    streams = data.get("streams", [])
    format_info = data.get("format", {})

    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if not v_stream:
        raise ValueError("Validation Failed [Stage: Video Stream]: No video stream found in output file.")

    if not a_stream:
        raise ValueError("Validation Failed [Stage: Audio Stream]: No audio stream found in output file.")

    w = int(v_stream.get("width", 0))
    h = int(v_stream.get("height", 0))
    codec = v_stream.get("codec_name", "").lower()
    pix_fmt = v_stream.get("pix_fmt", "").lower()
    audio_codec = a_stream.get("codec_name", "").lower()
    duration = float(format_info.get("duration", 0))

    print(f"Verified Specs: Width={w}, Height={h}, Video Codec={codec}, PixFmt={pix_fmt}, Duration={duration:.2f}s")

    if w != 720 or h != 1280:
        raise ValueError(f"Validation Failed [Stage: Resolution]: Resolution is {w}x{h}, expected 720x1280.")

    if codec != "h264":
        raise ValueError(f"Validation Failed [Stage: Video Codec]: Codec is {codec}, expected h264.")

    if pix_fmt != "yuv420p":
        raise ValueError(f"Validation Failed [Stage: Pixel Format]: Pixel format is {pix_fmt}, expected yuv420p.")

    if audio_codec != "aac":
        raise ValueError(f"Validation Failed [Stage: Audio Codec]: Codec is {audio_codec}, expected aac.")

    if duration <= 0:
        raise ValueError(f"Validation Failed [Stage: Duration]: Duration is {duration}s, expected > 0s.")

    video_duration = float(v_stream.get("duration", duration))
    audio_duration = float(a_stream.get("duration", duration))
    if abs(video_duration - audio_duration) > 0.5:
        raise ValueError(
            "Validation Failed [Stage: A/V Sync]: Video and audio durations differ "
            f"by {abs(video_duration - audio_duration):.2f}s."
        )

    validate_video_has_visible_content(file_path, duration)

    print("SUCCESS: Video validation passed all checks!\n")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        validate_video_file(target)
    else:
        print("Usage: python video_validator.py <path_to_video>")
