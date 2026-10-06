import json
import subprocess
from pathlib import Path

# Input video
video = Path("uploads/vidssave.com Data Structures Explained in Tamil _ DSA for Beginners 360P.mp4")

# Output folder
output_dir = Path("clips")
output_dir.mkdir(exist_ok=True)

# Load selected timestamps
with open("selected_segments.json", "r", encoding="utf-8") as file:
    segments = json.load(file)

# Create one clip for each selected segment
for i, segment in enumerate(segments, start=1):

    start = segment["start"]
    end = segment["end"]

    output_file = output_dir / f"clip_{i}.mp4"

    duration = end - start

    print(f"Creating clip {i}: {start:.2f}s → {end:.2f}s")

    subprocess.run([
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", str(video),
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        str(output_file)
    ], check=True)

print("\nAll clips created successfully!")