"""Semantic-Similarity-Only baseline for research evaluation."""

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import whisper
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_SECONDS = 10.0
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def create_candidate_segments(transcript_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group original Whisper segments into approximately 10-second candidates."""
    candidates = []
    chunk_start = None
    chunk_end = None
    chunk_text = []

    def append_chunk() -> None:
        if chunk_start is not None and chunk_end is not None and chunk_text:
            candidates.append(
                {
                    "start_time": chunk_start,
                    "end_time": chunk_end,
                    "text": " ".join(chunk_text),
                }
            )

    for segment in transcript_segments:
        text = str(segment.get("text", "")).strip()
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue

        if (
            not text
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
        ):
            continue

        if chunk_start is None:
            chunk_start = start
        chunk_end = end
        chunk_text.append(text)

        if chunk_end - chunk_start >= CHUNK_SECONDS:
            append_chunk()
            chunk_start = None
            chunk_end = None
            chunk_text = []

    append_chunk()
    return candidates


def select_non_overlapping_segments(
    candidates: list[dict[str, Any]],
    topic_embedding: Any,
    candidate_embeddings: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    """Rank by topic similarity and greedily choose non-overlapping candidates."""
    similarities = cosine_similarity(topic_embedding, candidate_embeddings)[0]
    ranked = sorted(
        zip(candidates, similarities),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    selected = []
    for candidate, similarity in ranked:
        overlaps = any(
            candidate["start_time"] < chosen["end_time"]
            and candidate["end_time"] > chosen["start_time"]
            for chosen in selected
        )
        if overlaps:
            continue

        selected.append(
            {
                **candidate,
                "similarity_score": float(similarity),
            }
        )
        if len(selected) >= top_k:
            break

    return selected


def extract_clip(input_path: Path, output_path: Path, start: float, end: float) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(input_path),
        "-t",
        f"{end - start:.6f}",
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
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise SystemExit("Error: ffmpeg was not found on PATH.") from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"Error: FFmpeg failed while creating {output_path} "
            f"(exit code {error.returncode})."
        ) from error


def run_semantic_only(
    input_path: Path,
    topic: str,
    output_dir: Path,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Transcribe, rank only by semantic similarity, and extract top clips."""
    input_path = input_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Whisper model (tiny)...")
    whisper_model = whisper.load_model("tiny")
    print("Transcribing source video...")
    transcript = whisper_model.transcribe(str(input_path), task="translate")
    candidates = create_candidate_segments(transcript.get("segments", []))
    if not candidates:
        raise SystemExit("Error: Whisper produced no usable transcript segments.")

    print(f"Created {len(candidates)} transcript candidate segment(s).")
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    topic_embedding = embedding_model.encode([topic])
    candidate_embeddings = embedding_model.encode(
        [candidate["text"] for candidate in candidates]
    )

    selected = select_non_overlapping_segments(
        candidates,
        topic_embedding,
        candidate_embeddings,
        top_k,
    )
    if not selected:
        raise SystemExit("Error: No non-overlapping candidate segments were selected.")

    metadata = []
    for index, segment in enumerate(selected, start=1):
        clip_id = f"clip_{index:03d}"
        start = segment["start_time"]
        end = segment["end_time"]
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise SystemExit(
                f"Error: Candidate has invalid original timestamps: {start} - {end}."
            )

        extract_clip(input_path, output_dir / f"{clip_id}.mp4", start, end)
        metadata.append(
            {
                "clip_id": clip_id,
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "duration": round(end - start, 3),
                "text": segment["text"],
                "similarity_score": round(segment["similarity_score"], 6),
            }
        )
        print(
            f"{clip_id}: {start:.2f}s - {end:.2f}s | "
            f"similarity={segment['similarity_score']:.4f}"
        )

    metadata_path = output_dir / "semantic_only_segments.json"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, ensure_ascii=False)
        metadata_file.write("\n")

    print(f"Selected {len(metadata)} clip(s). Metadata: {metadata_path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Semantic-Similarity-Only baseline: Whisper transcript chunks ranked "
            "only by MiniLM cosine similarity."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Source video.")
    parser.add_argument("--topic", required=True, help="Topic to search for.")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for clips and JSON metadata.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum number of non-overlapping clips to select (default: 3).",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Error: Input video does not exist: {args.input}")
    if not args.topic.strip():
        raise SystemExit("Error: Topic must not be empty.")
    if args.top_k < 1:
        raise SystemExit("Error: --top-k must be at least 1.")

    run_semantic_only(args.input, args.topic.strip(), args.output, args.top_k)


if __name__ == "__main__":
    main()
