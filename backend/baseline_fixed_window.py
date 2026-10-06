"""Create reproducible fixed-window clips as a research baseline."""

import argparse
import json
import math
import subprocess
from pathlib import Path

WINDOW_SECONDS = 30.0
MIN_FINAL_WINDOW_SECONDS = 10.0


def get_video_duration(input_path: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(input_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        duration = float(result.stdout.strip())
    except FileNotFoundError as error:
        raise SystemExit("Error: ffprobe was not found on PATH.") from error
    except (subprocess.CalledProcessError, ValueError) as error:
        raise SystemExit(
            f"Error: Could not determine the duration of {input_path}: {error}"
        ) from error

    if not math.isfinite(duration) or duration <= 0:
        raise SystemExit(f"Error: Invalid video duration reported: {duration}.")
    return duration


def create_fixed_window_clips(input_path: Path, output_dir: Path) -> list[dict]:
    """Create clips using fixed 30-second windows; no content analysis is used."""
    input_path = input_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_duration = get_video_duration(input_path)

    segments = []
    start_time = 0.0
    while start_time < video_duration:
        end_time = min(start_time + WINDOW_SECONDS, video_duration)
        clip_duration = end_time - start_time

        # The Fixed-Window Baseline discards a final remainder shorter than 10s.
        if (
            clip_duration < WINDOW_SECONDS
            and clip_duration < MIN_FINAL_WINDOW_SECONDS
        ):
            print(
                f"Skipping final {clip_duration:.2f}s remainder "
                f"(minimum is {MIN_FINAL_WINDOW_SECONDS:.0f}s)."
            )
            break

        clip_id = f"clip_{len(segments) + 1:03d}"
        clip_path = output_dir / f"{clip_id}.mp4"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_time:.6f}",
            "-i",
            str(input_path),
            "-t",
            f"{clip_duration:.6f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-sn",
            "-dn",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(clip_path),
        ]
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as error:
            raise SystemExit("Error: ffmpeg was not found on PATH.") from error
        except subprocess.CalledProcessError as error:
            raise SystemExit(
                f"Error: FFmpeg failed while creating {clip_path} "
                f"(exit code {error.returncode})."
            ) from error

        segments.append(
            {
                "clip_id": clip_id,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "duration": round(clip_duration, 3),
                "source_video": str(input_path),
            }
        )
        print(
            f"Created {clip_path.name}: "
            f"{start_time:.2f}s - {end_time:.2f}s"
        )
        start_time = end_time

    metadata_path = output_dir / "fixed_window_segments.json"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(segments, metadata_file, indent=2, ensure_ascii=False)
        metadata_file.write("\n")

    print(f"Created {len(segments)} fixed-window clip(s).")
    print(f"Metadata: {metadata_path}")
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create 30-second Fixed-Window Baseline clips. "
            "No transcription, AI scoring, or content analysis is used."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Source video.")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory for baseline clips and JSON metadata.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Error: Input video does not exist: {args.input}")

    create_fixed_window_clips(args.input, args.output)


if __name__ == "__main__":
    main()
