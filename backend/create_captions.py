import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

TRANSCRIPT_FILE = BASE_DIR / "transcript_english" / "pipeline_transcript.json"
SRT_FILE = BASE_DIR / "captions.srt"


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)

    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as file:
    transcript = json.load(file)


with open(SRT_FILE, "w", encoding="utf-8") as file:

    for index, segment in enumerate(transcript["segments"], start=1):

        start = format_time(segment["start"])
        end = format_time(segment["end"])
        text = segment["text"].strip()

        file.write(f"{index}\n")
        file.write(f"{start} --> {end}\n")
        file.write(f"{text}\n\n")


print("Captions created successfully!")
print("Saved to:", SRT_FILE)