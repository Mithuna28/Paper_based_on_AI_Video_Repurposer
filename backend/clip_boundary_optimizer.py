import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


MIN_DURATION = 10
MAX_DURATION = 30


def optimize_segment(segment, transcript_segments):
    start = segment["start"]
    end = segment["end"]

    # If already within the ideal range
    if MIN_DURATION <= (end - start) <= MAX_DURATION:
        return segment

    # Find Whisper segments that overlap the selected segment
    overlapping = []

    for item in transcript_segments:
        if item["end"] > start and item["start"] < end:
            overlapping.append(item)

    if not overlapping:
        return segment

    # Start at the first complete Whisper segment
    new_start = overlapping[0]["start"]

    # Build a clip up to MAX_DURATION
    new_end = new_start

    for item in overlapping:
        if item["end"] - new_start <= MAX_DURATION:
            new_end = item["end"]
        else:
            break

    # Make sure the clip isn't too short
    if new_end - new_start < MIN_DURATION:
        return segment

    optimized = segment.copy()
    optimized["start"] = round(new_start, 2)
    optimized["end"] = round(new_end, 2)

    return optimized


def optimize_boundaries():
    selected_file = BASE_DIR / "selected_segments.json"
    transcript_file = (
        BASE_DIR
        / "transcript_english"
        / "pipeline_transcript.json"
    )

    output_file = BASE_DIR / "optimized_segments.json"

    with open(selected_file, "r", encoding="utf-8") as file:
        selected_segments = json.load(file)

    with open(transcript_file, "r", encoding="utf-8") as file:
        transcript = json.load(file)

    transcript_segments = transcript["segments"]

    optimized_segments = []

    for segment in selected_segments:
        optimized = optimize_segment(
            segment,
            transcript_segments
        )

        optimized_segments.append(optimized)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            optimized_segments,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n=== SMART CLIP BOUNDARY OPTIMIZATION ===")

    for index, segment in enumerate(optimized_segments, 1):
        duration = segment["end"] - segment["start"]

        print(
            f"{index}. "
            f"{segment['start']:.2f}s - "
            f"{segment['end']:.2f}s "
            f"({duration:.2f}s)"
        )


if __name__ == "__main__":
    optimize_boundaries()