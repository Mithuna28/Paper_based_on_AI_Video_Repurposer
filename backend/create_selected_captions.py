import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

TRANSCRIPT_FILE = BASE_DIR / "transcript_english" / "pipeline_transcript.json"
SELECTED_FILE = BASE_DIR / "selected_segments.json"
OUTPUT_FILE = BASE_DIR / "selected_captions.srt"

def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))

    if milliseconds >= 1000:
        secs += 1
        milliseconds = 0

    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def generate_srt_captions(transcript_path: Path, selected_path: Path, output_path: Path):
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    with open(selected_path, "r", encoding="utf-8") as f:
        selected_segments = json.load(f)

    srt_entries = []
    new_time = 0.0

    for selected in selected_segments:
        original_start = selected["start"]
        original_end = selected["end"]
        clip_duration = original_end - original_start

        for segment in transcript.get("segments", []):
            seg_start = segment["start"]
            seg_end = segment["end"]

            if seg_end > original_start and seg_start < original_end:
                new_start = max(seg_start, original_start) - original_start + new_time
                new_end = min(seg_end, original_end) - original_start + new_time
                if new_end > new_start:
                    srt_entries.append({
                        "start": new_start,
                        "end": new_end,
                        "text": segment["text"].strip()
                    })

        new_time += clip_duration

    with open(output_path, "w", encoding="utf-8") as f:
        for index, entry in enumerate(srt_entries, start=1):
            f.write(f"{index}\n")
            f.write(
                f"{format_time(entry['start'])} --> "
                f"{format_time(entry['end'])}\n"
            )
            f.write(f"{entry['text']}\n\n")

    print(f"Selected captions generated successfully ({len(srt_entries)} entries) -> {output_path}")

if __name__ == "__main__":
    generate_srt_captions(TRANSCRIPT_FILE, SELECTED_FILE, OUTPUT_FILE)