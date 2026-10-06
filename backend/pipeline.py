import sys
import os
import json
import math
import shutil
import subprocess
from pathlib import Path

import whisper
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from highlight_ranker import calculate_highlight_score
from clip_boundary_optimizer import optimize_segment
from duplicate_remover import remove_duplicate_segments
from scene_segmentation import detect_scenes
from final_selection import select_best_clips
from content_analyzer import ContentAnalyzer
from video_validator import validate_video_file, validate_video_has_visible_content
from create_selected_captions import generate_srt_captions

MIN_SHORT_DURATION = 60.0
MAX_SHORT_DURATION = 120.0
SIMILARITY_THRESHOLD = 0.35


def run_ffmpeg(command, stage):
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise SystemExit(f"Error: FFmpeg was not found while {stage}.") from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"Error: FFmpeg failed while {stage} (exit code {error.returncode})."
        ) from error


def get_video_duration(video_path):
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        duration = float(result.stdout.strip())
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("FFprobe returned a non-positive duration.")
        return duration
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as error:
        raise SystemExit(
            f"Error: Could not determine the duration of {video_path}: {error}"
        ) from error


def write_pipeline_json(filename, data):
    output_file = BASE_DIR / filename
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


print("=========================================")
print("  AI Video Repurposer Pipeline Running   ")
print("=========================================")

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
AUDIO_DIR = BASE_DIR / "audio"
TRANSCRIPT_DIR = BASE_DIR / "transcript_english"
CLIP_DIR = BASE_DIR / "clips"

AUDIO_DIR.mkdir(exist_ok=True)
TRANSCRIPT_DIR.mkdir(exist_ok=True)
CLIP_DIR.mkdir(exist_ok=True)

if len(sys.argv) < 3:
    print("Error: Required arguments missing. Usage: python pipeline.py <video_filename> <topic>")
    sys.exit(1)

video_filename = sys.argv[1]
topic = sys.argv[2]

VIDEO_FILE = UPLOAD_DIR / video_filename
AUDIO_FILE = AUDIO_DIR / "pipeline_audio.wav"
TRANSCRIPT_FILE = TRANSCRIPT_DIR / "pipeline_transcript.json"

print(f"Target Video : {VIDEO_FILE}")
print(f"Audio Output : {AUDIO_FILE}")
print(f"User Topic   : {topic}")

if not VIDEO_FILE.is_file():
    raise SystemExit(f"Error: Source video does not exist: {VIDEO_FILE}")

validate_video_has_visible_content(VIDEO_FILE)
source_duration = get_video_duration(VIDEO_FILE)

# ---------------------------------------------------------
# Step 1: Extract Audio
# ---------------------------------------------------------
print("\n[Step 1/10] Extracting audio from video...")

run_ffmpeg([
    "ffmpeg", "-y",
    "-i", str(VIDEO_FILE),
    "-vn",
    "-ac", "1",
    "-ar", "16000",
    "-c:a", "pcm_s16le",
    str(AUDIO_FILE)
], "extracting audio")

print("Audio extraction completed successfully!")

# ---------------------------------------------------------
# Step 2: Whisper Transcription
# ---------------------------------------------------------
print("\n[Step 2/10] Loading Whisper model (tiny)...")
whisper_model = whisper.load_model("tiny")

print("Transcribing audio...")
transcript_result = whisper_model.transcribe(
    str(AUDIO_FILE),
    task="translate"
)

with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
    json.dump(transcript_result, f, ensure_ascii=False, indent=4)

print(f"Whisper transcription completed! Saved to: {TRANSCRIPT_FILE}")

# ---------------------------------------------------------
# Step 3: MiniLM Semantic Search & Clip Selection
# ---------------------------------------------------------
print("\n[Step 3/10] Running MiniLM semantic search...")
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

segments = transcript_result.get("segments", [])
if not segments:
    raise SystemExit("Error: Whisper did not find any transcript segments.")

validated_segments = []
for index, segment in enumerate(segments, start=1):
    start = float(segment["start"])
    end = float(segment["end"])
    if (
        not math.isfinite(start)
        or not math.isfinite(end)
        or start < 0
        or end <= start
        or end > source_duration
    ):
        raise SystemExit(
            f"Error: Whisper segment {index} has invalid timestamps "
            f"({start} - {end}) for a {source_duration:.2f}s video."
        )
    validated_segment = dict(segment)
    validated_segment["start"] = start
    validated_segment["end"] = end
    validated_segments.append(validated_segment)
segments = validated_segments

chunks = []
chunk_text = ""
chunk_start = None
chunk_end = None

for segment in segments:
    segment_text = segment.get("text", "").strip()
    if not segment_text:
        continue

    if chunk_start is None:
        chunk_start = segment["start"]

    chunk_text += " " + segment_text
    chunk_end = segment["end"]

    if chunk_end - chunk_start >= 10:
        chunks.append({
            "start": chunk_start,
            "end": chunk_end,
            "text": chunk_text.strip()
        })
        chunk_text = ""
        chunk_start = None
        chunk_end = None

if chunk_text:
    chunks.append({
        "start": chunk_start,
        "end": chunk_end,
        "text": chunk_text.strip()
    })

texts = [chunk["text"] for chunk in chunks]
if not texts:
    raise SystemExit("Error: Whisper returned no usable transcript text chunks.")

query_embedding = semantic_model.encode([topic])
text_embeddings = semantic_model.encode(texts)

scores = cosine_similarity(query_embedding, text_embeddings)[0]

top_score = max(scores) if len(scores) > 0 else 0
print(f"Top similarity score: {top_score:.3f}")

results = []
for i, score in enumerate(scores):
    results.append({
        "start": chunks[i]["start"],
        "end": chunks[i]["end"],
        "text": chunks[i]["text"],
        "score": float(score)
    })

filtered_segments = [
    segment for segment in results
    if segment["score"] >= SIMILARITY_THRESHOLD
]
if not filtered_segments:
    raise SystemExit(
        f"Error: No transcript segments were relevant to '{topic}' "
        f"(similarity threshold {SIMILARITY_THRESHOLD:.2f})."
    )

filtered_segments.sort(key=lambda segment: segment["start"])
merged_segments = []
for segment in filtered_segments:
    current = dict(segment)
    if merged_segments and current["start"] <= merged_segments[-1]["end"] + 1.5:
        previous = merged_segments[-1]
        previous["end"] = max(previous["end"], current["end"])
        previous["text"] = f"{previous['text']} {current['text']}".strip()
        previous["score"] = max(previous["score"], current["score"])
    else:
        merged_segments.append(current)

for segment in merged_segments:
    segment["highlight_score"] = calculate_highlight_score(segment)

ranked_segments = sorted(
    merged_segments,
    key=lambda segment: segment["highlight_score"],
    reverse=True
)
write_pipeline_json("selected_segments.json", merged_segments)
write_pipeline_json("ranked_segments.json", ranked_segments)

print("\n=== SMART HIGHLIGHT RANKING ===")
for index, segment in enumerate(ranked_segments, start=1):
    print(
        f"{index}. {segment['start']:.2f}s - {segment['end']:.2f}s | "
        f"MiniLM score={segment['score']:.4f} | "
        f"highlight score={segment['highlight_score']:.4f}"
    )

optimized_segments = [
    optimize_segment(segment, segments)
    for segment in ranked_segments
]
for segment in optimized_segments:
    segment["highlight_score"] = calculate_highlight_score(segment)

valid_optimized_segments = []
for segment in optimized_segments:
    start = segment["start"]
    end = segment["end"]
    duration = end - start
    if (
        not math.isfinite(start)
        or not math.isfinite(end)
        or start < 0
        or end > source_duration
        or duration < 0.5
        or duration > MAX_SHORT_DURATION
    ):
        print(
            f"Warning: discarded invalid optimized segment "
            f"{start:.2f}s - {end:.2f}s ({duration:.2f}s)."
        )
        continue
    valid_optimized_segments.append(segment)

optimized_segments = valid_optimized_segments
optimized_segments.sort(
    key=lambda segment: segment["highlight_score"],
    reverse=True,
)
if not optimized_segments:
    raise SystemExit(
        "Error: Boundary optimization produced no valid segments within "
        f"the {MAX_SHORT_DURATION:.0f}s per-clip limit."
    )
write_pipeline_json("optimized_segments.json", optimized_segments)

print("\n=== SMART CLIP BOUNDARY OPTIMIZATION ===")
for index, segment in enumerate(optimized_segments, start=1):
    print(
        f"{index}. {segment['start']:.2f}s - {segment['end']:.2f}s "
        f"({segment['end'] - segment['start']:.2f}s)"
    )

unique_segments = remove_duplicate_segments(
    optimized_segments,
    semantic_model,
)
write_pipeline_json("unique_segments.json", unique_segments)
print("\n=== DUPLICATE REMOVAL ===")
print(f"Original clips: {len(optimized_segments)}")
print(f"Unique clips: {len(unique_segments)}")
print(f"Removed clips: {len(optimized_segments) - len(unique_segments)}")
if not unique_segments:
    raise SystemExit("Error: No unique topic-relevant segments could be selected.")

scene_segments = detect_scenes(VIDEO_FILE)
write_pipeline_json("scene_segments.json", scene_segments)
print("\n=== SCENE SEGMENTATION ===")
print(f"Scenes detected: {len(scene_segments)}")

selected_segments = select_best_clips(
    unique_segments,
    scene_segments,
    max_duration=MAX_SHORT_DURATION,
)
selected_segments.sort(key=lambda segment: segment["start"])
if not selected_segments:
    raise SystemExit("Error: No valid topic-relevant segments could be selected.")
write_pipeline_json("final_segments.json", selected_segments)

selected_duration = sum(
    segment["end"] - segment["start"]
    for segment in selected_segments
)
SELECTED_FILE = BASE_DIR / "selected_segments.json"
SELECTED_FILE = BASE_DIR / "selected_segments.json"
write_pipeline_json("selected_segments.json", selected_segments)

print(f"Selected duration: {selected_duration:.2f} seconds")
if selected_duration < MIN_SHORT_DURATION:
    print(
        f"Warning: only {selected_duration:.2f}s of relevant content was found; "
        f"the target minimum is {MIN_SHORT_DURATION:.0f}s. No unrelated content was added."
    )

print(f"Selected {len(selected_segments)} relevant segment(s), in chronological order:")
for s in selected_segments:
    print(
        f"  [{s['start']:.2f}s - {s['end']:.2f}s] "
        f"Score: {s['score']:.3f} | Highlight: {s['highlight_score']:.4f} | "
        f"Text: {s['text'][:60]}..."
    )

# ---------------------------------------------------------
# Step 4: Create Individual Video Clips
# ---------------------------------------------------------
print("\n[Step 4/10] Extracting video clips...")
for f in CLIP_DIR.glob("clip_*.mp4"):
    f.unlink()

clip_paths = []
for i, segment in enumerate(selected_segments, start=1):
    start = segment["start"]
    end = segment["end"]
    duration = max(0.5, end - start)
    output_clip = CLIP_DIR / f"clip_{i}.mp4"

    print(f"  Extracting clip {i}: {start:.2f}s -> {end:.2f}s ({duration:.2f}s)")

    run_ffmpeg([
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(VIDEO_FILE),
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_clip)
    ], f"extracting clip {i}")

    clip_paths.append(output_clip)

print(f"Video clips created successfully! Count: {len(clip_paths)}")

# ---------------------------------------------------------
# Step 5: Combine Video Clips
# ---------------------------------------------------------
COMBINED_VIDEO = BASE_DIR / "combined_pipeline.mp4"
print("\n[Step 5/10] Concatenating video clips...")

CONCAT_TXT = BASE_DIR / "pipeline_clips.txt"
with open(CONCAT_TXT, "w", encoding="utf-8") as f:
    for clip_p in clip_paths:
        f.write(f"file '{clip_p.resolve()}'\n")

run_ffmpeg([
    "ffmpeg", "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", str(CONCAT_TXT),
    "-c:v", "libx264",
    "-c:a", "aac",
    str(COMBINED_VIDEO)
], "combining selected clips")

print(f"Clips combined successfully -> {COMBINED_VIDEO}")

# ---------------------------------------------------------
# Step 6: Convert to 9:16 Vertical Video (Preserving Source Visibility)
# ---------------------------------------------------------
VERTICAL_VIDEO = BASE_DIR / "vertical_pipeline.mp4"
print("\n[Step 6/10] Converting to 720x1280 (9:16 vertical)...")

# Safe vertical filter graph:
# Split into background (blurred & filled to 720x1280) and main video (scaled to fit 720x1280 aspect ratio).
vertical_filter = (
    "[0:v]split=2[background][foreground];"
    "[background]scale=720:1280:force_original_aspect_ratio=increase,"
    "crop=720:1280,boxblur=20:10,setsar=1[background_blurred];"
    "[foreground]scale=720:1280:force_original_aspect_ratio=decrease:"
    "force_divisible_by=2,setsar=1[foreground_fit];"
    "[background_blurred][foreground_fit]overlay=(W-w)/2:(H-h)/2,"
    "format=yuv420p[vertical]"
)

run_ffmpeg([
    "ffmpeg", "-y",
    "-i", str(COMBINED_VIDEO),
    "-filter_complex", vertical_filter,
    "-map", "[vertical]",
    "-map", "0:a?",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-c:a", "aac",
    "-b:a", "128k",
    "-shortest",
    "-movflags", "+faststart",
    str(VERTICAL_VIDEO)
], "converting the video to 9:16")

print(f"Vertical video created successfully -> {VERTICAL_VIDEO}")
validate_video_has_visible_content(VERTICAL_VIDEO)

# ---------------------------------------------------------
# Step 7: Create Selected SRT Captions
# ---------------------------------------------------------
CAPTIONS_FILE = BASE_DIR / "selected_captions.srt"
print("\n[Step 7/10] Generating synchronized SRT captions...")
generate_srt_captions(TRANSCRIPT_FILE, SELECTED_FILE, CAPTIONS_FILE)

# ---------------------------------------------------------
# Step 8: Content Analysis & Effect Chain Generation
# ---------------------------------------------------------
print("\n[Step 8/10] Analyzing transcript for content-aware animations & effects...")
analyzer = ContentAnalyzer(topic=topic, font_path="C\\:/Windows/Fonts/arialbd.ttf")
analyzed_effects = analyzer.analyze_segments(selected_segments)
effect_filter_string = analyzer.build_ffmpeg_filter_chain(analyzed_effects)

print(f"Generated {len(analyzed_effects)} content-aware effect(s):")
for eff in analyzed_effects:
    print(f"  [{eff['type']}] ({eff['start']:.2f}s - {eff['end']:.2f}s) Text: {eff['text']}")

# ---------------------------------------------------------
# Step 9: Render Final Video with Subtitles + Content-Aware Effects
# ---------------------------------------------------------
FINAL_VIDEO = BASE_DIR / "final_pipeline.mp4"
print("\n[Step 9/10] Rendering final edited video with subtitles and effects...")

# Windows subtitle filter path escaping
srt_path_escaped = str(CAPTIONS_FILE.resolve()).replace("\\", "/").replace(":", "\\:")

subtitle_filter = (
    f"subtitles=filename='{srt_path_escaped}':"
    "force_style='PlayResX=720,PlayResY=1280,Fontname=Arial,Fontsize=30,"
    "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,"
    "Outline=1,Shadow=1,Alignment=2,MarginV=120'"
)

zoom_filter = (
    "scale=iw*1.05:ih*1.05,"
    "crop=720:1280:(in_w-720)/2:(in_h-1280)/2"
)
full_filter_chain = f"{zoom_filter},{subtitle_filter}"
if effect_filter_string:
    full_filter_chain += f",{effect_filter_string}"

run_ffmpeg([
    "ffmpeg", "-y",
    "-i", str(VERTICAL_VIDEO),
    "-vf", full_filter_chain,
    "-map", "0:v:0",
    "-map", "0:a?",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-c:a", "aac",
    "-b:a", "128k",
    "-shortest",
    "-movflags", "+faststart",
    "-t", str(MAX_SHORT_DURATION),
    str(FINAL_VIDEO)
], "rendering the final video")

print(f"Final video rendered -> {FINAL_VIDEO}")

# ---------------------------------------------------------
# Step 10: Automatic Validation
# ---------------------------------------------------------
print("\n[Step 10/10] Performing automatic quality validation...")
if not FINAL_VIDEO.is_file():
    raise SystemExit(f"Error: Final video was not created: {FINAL_VIDEO}")

validate_video_file(FINAL_VIDEO)
final_duration = get_video_duration(FINAL_VIDEO)
print(f"Final video duration: {final_duration:.2f} seconds")
if final_duration < MIN_SHORT_DURATION:
    print(
        f"Warning: final video is shorter than {MIN_SHORT_DURATION:.0f}s because "
        "there was not enough relevant content."
    )
if final_duration > MAX_SHORT_DURATION + 0.5:
    raise SystemExit(
        f"Error: Final video duration {final_duration:.2f}s exceeds the "
        f"{MAX_SHORT_DURATION:.0f}s maximum."
    )

print("=========================================")
print("  PIPELINE EXECUTED AND VALIDATED OK!    ")
print("=========================================")