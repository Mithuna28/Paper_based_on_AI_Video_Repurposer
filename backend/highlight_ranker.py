import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def calculate_highlight_score(segment):
    semantic_score = segment["score"]

    duration = segment["end"] - segment["start"]

    # Prefer useful medium-length clips
    if 5 <= duration <= 30:
        duration_score = 1.0
    elif duration < 5:
        duration_score = 0.5
    else:
        duration_score = 0.7

    final_score = (
        0.75 * semantic_score
        + 0.25 * duration_score
    )

    return round(final_score, 4)


def rank_highlights():
    input_file = BASE_DIR / "selected_segments.json"
    output_file = BASE_DIR / "ranked_segments.json"

    with open(input_file, "r", encoding="utf-8") as file:
        segments = json.load(file)

    for segment in segments:
        segment["highlight_score"] = calculate_highlight_score(segment)

    segments.sort(
        key=lambda x: x["highlight_score"],
        reverse=True
    )

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            segments,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n=== SMART HIGHLIGHT RANKING ===")

    for index, segment in enumerate(segments, start=1):
        print(
            f"{index}. "
            f"{segment['start']:.1f}s - "
            f"{segment['end']:.1f}s | "
            f"score={segment['highlight_score']}"
        )

    return segments


if __name__ == "__main__":
    rank_highlights()