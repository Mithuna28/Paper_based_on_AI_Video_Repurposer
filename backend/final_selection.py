import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "unique_segments.json"
OUTPUT_FILE = BASE_DIR / "final_segments.json"

MAX_CLIPS = 3

def select_best_clips(segments=None, scenes=None, max_duration=None):
    if segments is None:
        with open(INPUT_FILE, "r", encoding="utf-8") as file:
            segments = json.load(file)

    scene_containment = {}
    for segment in segments:
        scene_containment[id(segment)] = any(
            scene["start"] <= segment["start"]
            and segment["end"] <= scene["end"]
            for scene in (scenes or [])
        )

    segments = sorted(
        segments,
        key=lambda x: (
            x.get("highlight_score", x.get("score", 0)),
            scene_containment[id(x)],
        ),
        reverse=True
    )

    selected = []
    selected_duration = 0.0
    for segment in segments:
        duration = segment["end"] - segment["start"]
        if (
            max_duration is not None
            and selected_duration + duration > max_duration
        ):
            continue

        selected.append(segment)
        selected_duration += duration
        if len(selected) == MAX_CLIPS:
            break

    # Final chronological ordering
    selected.sort(key=lambda x: x["start"])

    if scenes is None:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
            json.dump(selected, file, indent=4, ensure_ascii=False)

    print("\n=== FINAL BEST CLIP SELECTION ===")
    print(f"Selected clips: {len(selected)}")

    for i, clip in enumerate(selected, 1):
        print(
            f"{i}. {clip['start']:.2f}s - "
            f"{clip['end']:.2f}s | "
            f"score={clip.get('highlight_score', clip.get('score', 0)):.4f}"
        )

    return selected

if __name__ == "__main__":
    select_best_clips()
