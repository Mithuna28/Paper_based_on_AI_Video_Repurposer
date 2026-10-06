"""Evaluate Fixed Window, Semantic Only, and Proposed video clips."""

import argparse
import csv
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import whisper
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
METHOD_METADATA = {
    "fixed_window": (
        "fixed_window_segments.json",
        "start_time",
        "end_time",
    ),
    "semantic_only": (
        "semantic_only_segments.json",
        "start_time",
        "end_time",
    ),
    "proposed": (
        "final_segments.json",
        "start",
        "end",
    ),
}
METADATA_FALLBACKS = (
    ("final_segments.json", "start", "end"),
    ("selected_segments.json", "start", "end"),
    ("semantic_only_segments.json", "start_time", "end_time"),
    ("fixed_window_segments.json", "start_time", "end_time"),
)
CSV_FIELDS = (
    "video",
    "topic",
    "method",
    "semantic_relevance",
    "content_coverage",
    "redundancy",
)


def get_video_duration(video_path: Path) -> float:
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
                str(video_path),
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
            f"Error: Could not determine the duration of {video_path}: {error}"
        ) from error

    if not math.isfinite(duration) or duration <= 0:
        raise SystemExit(f"Error: Invalid video duration reported: {duration}.")
    return duration


def natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def find_clip_files(directory: Path) -> list[Path]:
    clips = sorted(
        (
            path
            for path in directory.rglob("clip_*.mp4")
            if path.is_file()
        ),
        key=natural_key,
    )
    if not clips:
        raise SystemExit(
            f"Error: No clip_*.mp4 files were found under {directory}."
        )
    return clips


def find_metadata(
    directory: Path,
    method: str,
) -> tuple[Path, list[dict[str, Any]], str, str]:
    preferred_name, preferred_start, preferred_end = METHOD_METADATA[method]
    search_dirs = []
    for candidate_dir in (directory, *directory.parents[:3]):
        if candidate_dir not in search_dirs:
            search_dirs.append(candidate_dir)

    choices = [(preferred_name, preferred_start, preferred_end)]
    choices.extend(
        choice for choice in METADATA_FALLBACKS if choice[0] != preferred_name
    )
    for search_dir in search_dirs:
        for filename, start_key, end_key in choices:
            metadata_path = search_dir / filename
            if metadata_path.is_file():
                with metadata_path.open("r", encoding="utf-8") as file:
                    records = json.load(file)
                if not isinstance(records, list):
                    raise SystemExit(
                        f"Error: Expected a JSON list in {metadata_path}."
                    )
                return metadata_path, records, start_key, end_key

    raise SystemExit(
        f"Error: Could not find source-timestamp metadata for {method} "
        f"under {directory} or its parent directories."
    )


def clip_number(value: Any) -> int | None:
    match = re.search(r"(\d+)$", str(value))
    return int(match.group(1)) if match else None


def load_clip_intervals(
    directory: Path,
    method: str,
    video_duration: float,
    source_video_path: Path,
) -> tuple[list[Path], list[tuple[float, float]]]:
    clips = find_clip_files(directory)
    metadata_path, records, start_key, end_key = find_metadata(directory, method)

    if len(clips) != len(records):
        raise SystemExit(
            f"Error: {method} has {len(clips)} clip file(s) but "
            f"{len(records)} records in {metadata_path}."
        )

    records_by_id = {
        clip_number(record["clip_id"]): record
        for record in records
        if isinstance(record, dict) and record.get("clip_id") is not None
    }
    intervals = []
    for index, clip_path in enumerate(clips):
        clip_id = clip_number(clip_path.stem)
        record = records_by_id.get(clip_id) if records_by_id else records[index]
        if record is None:
            record = records[index]
        try:
            start = float(record[start_key])
            end = float(record[end_key])
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(
                f"Error: Missing numeric source timestamps for {clip_path.name} "
                f"in {metadata_path}."
            ) from error

        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or end > video_duration + 0.5
        ):
            raise SystemExit(
                f"Error: Invalid source interval for {clip_path.name}: "
                f"{start} - {end} (source duration {video_duration:.2f}s)."
            )

        metadata_source = record.get("source_video")
        if metadata_source:
            metadata_source_path = Path(metadata_source).resolve()
            if metadata_source_path != source_video_path.resolve():
                raise SystemExit(
                    f"Error: {metadata_path} refers to {metadata_source_path}, "
                    f"not the requested source {source_video_path.resolve()}."
                )
        intervals.append((start, end))

    return clips, intervals


def merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def total_interval_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def calculate_content_coverage(
    relevant_intervals: list[tuple[float, float]],
    selected_intervals: list[tuple[float, float]],
) -> float:
    relevant_union = merge_intervals(relevant_intervals)
    total_relevant_duration = sum(
        end - start for start, end in relevant_union
    )
    if total_relevant_duration <= 0:
        return 0.0

    covered = []
    selected_union = merge_intervals(selected_intervals)
    for relevant_start, relevant_end in relevant_union:
        for selected_start, selected_end in selected_union:
            overlap_start = max(relevant_start, selected_start)
            overlap_end = min(relevant_end, selected_end)
            if overlap_end > overlap_start:
                covered.append((overlap_start, overlap_end))

    covered_duration = total_interval_duration(covered)
    return min(1.0, covered_duration / total_relevant_duration)


def transcribe_text(model: Any, video_path: Path) -> str:
    result = model.transcribe(str(video_path), task="translate")
    segments = result.get("segments", [])
    text = " ".join(
        str(segment.get("text", "")).strip()
        for segment in segments
        if str(segment.get("text", "")).strip()
    )
    return text or str(result.get("text", "")).strip()


def evaluate_method(
    clip_paths: list[Path],
    intervals: list[tuple[float, float]],
    clip_texts: list[str],
    clip_embeddings: Any,
    topic_embedding: Any,
    relevant_source_intervals: list[tuple[float, float]],
    threshold: float,
) -> dict[str, float | int]:
    clip_scores = [0.0] * len(clip_paths)
    nonempty_indices = [
        index for index, text in enumerate(clip_texts) if text.strip()
    ]
    nonempty_embeddings = [
        clip_embeddings[index] for index in nonempty_indices
    ]
    if nonempty_indices:
        similarities = cosine_similarity(
            topic_embedding,
            nonempty_embeddings,
        )[0]
        for index, similarity in zip(nonempty_indices, similarities):
            clip_scores[index] = float(similarity)

    if len(nonempty_indices) >= 2:
        pairwise = cosine_similarity(nonempty_embeddings)
        upper_triangle = pairwise[np.triu_indices(len(nonempty_indices), k=1)]
        redundancy = float(upper_triangle.mean())
    else:
        redundancy = 0.0

    semantic_relevance = (
        sum(clip_scores) / len(clip_scores) if clip_scores else 0.0
    )
    return {
        "semantic_relevance": round(semantic_relevance, 6),
        "content_coverage": round(
            calculate_content_coverage(
                relevant_source_intervals,
                intervals,
            ),
            6,
        ),
        "redundancy": round(redundancy, 6),
        "clip_count": len(clip_paths),
        "relevant_source_duration": round(
            total_interval_duration(relevant_source_intervals),
            3,
        ),
        "relevance_threshold": threshold,
    }


def evaluate(
    video: Path,
    topic: str,
    method_dirs: dict[str, Path],
    output_dir: Path,
    threshold: float = 0.35,
    whisper_model_name: str = "tiny",
) -> dict[str, Any]:
    source_video = video.resolve()
    source_duration = get_video_duration(source_video)
    loaded_clips = {
        method: load_clip_intervals(
            directory.resolve(),
            method,
            source_duration,
            source_video,
        )
        for method, directory in method_dirs.items()
    }

    print(f"Loading Whisper model ({whisper_model_name})...")
    whisper_model = whisper.load_model(whisper_model_name)
    print("Transcribing source video for coverage reference...")
    source_transcript = whisper_model.transcribe(
        str(source_video),
        task="translate",
    )
    source_segments = [
        segment
        for segment in source_transcript.get("segments", [])
        if str(segment.get("text", "")).strip()
    ]
    if not source_segments:
        raise SystemExit("Error: Whisper found no usable source transcript.")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    topic_embedding = embedding_model.encode([topic])
    source_texts = [str(segment["text"]).strip() for segment in source_segments]
    source_embeddings = embedding_model.encode(source_texts)
    source_scores = cosine_similarity(topic_embedding, source_embeddings)[0]
    relevant_source_intervals = []
    for segment, similarity in zip(source_segments, source_scores):
        start = float(segment["start"])
        end = float(segment["end"])
        if (
            float(similarity) >= threshold
            and math.isfinite(start)
            and math.isfinite(end)
            and 0 <= start < end <= source_duration + 0.5
        ):
            relevant_source_intervals.append((start, min(end, source_duration)))

    print(
        f"Relevant source transcript segments (similarity >= {threshold:.3f}): "
        f"{len(relevant_source_intervals)}; "
        f"union duration={total_interval_duration(relevant_source_intervals):.2f}s"
    )

    texts_by_method: dict[str, list[str]] = {}
    all_nonempty_texts = []
    for method, (clip_paths, _) in loaded_clips.items():
        texts = []
        for clip_path in clip_paths:
            print(f"Transcribing {method} clip: {clip_path.name}")
            text = transcribe_text(whisper_model, clip_path)
            texts.append(text)
            if text:
                all_nonempty_texts.append(text)
        texts_by_method[method] = texts

    if not all_nonempty_texts:
        raise SystemExit("Error: Whisper found no speech in any generated clip.")

    all_embeddings = embedding_model.encode(all_nonempty_texts)
    embedding_by_text = {
        text: embedding
        for text, embedding in zip(all_nonempty_texts, all_embeddings)
    }

    methods = {}
    for method, (clip_paths, intervals) in loaded_clips.items():
        texts = texts_by_method[method]
        embeddings = [
            embedding_by_text.get(text, [0.0] * len(all_embeddings[0]))
            for text in texts
        ]
        methods[method] = evaluate_method(
            clip_paths,
            intervals,
            texts,
            embeddings,
            topic_embedding,
            relevant_source_intervals,
            threshold,
        )

    results = {
        "video": str(source_video),
        "topic": topic,
        "embedding_model": EMBEDDING_MODEL,
        "whisper_model": whisper_model_name,
        "relevance_threshold": threshold,
        "relevance_threshold_rule": (
            "A source transcript segment is relevant when its cosine "
            "similarity with the topic embedding is greater than or equal "
            "to this threshold."
        ),
        "metric_formulas": {
            "semantic_relevance": (
                "Mean topic-to-clip-transcript cosine similarity; clips "
                "without recognized speech score 0."
            ),
            "content_coverage": (
                "Duration of the union of relevant source-transcript "
                "intervals overlapped by selected source intervals divided "
                "by the duration of the union of all relevant intervals."
            ),
            "redundancy": (
                "Mean pairwise cosine similarity between non-empty selected "
                "clip transcript embeddings; 0 when fewer than two are usable."
            ),
        },
        "methods": methods,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation_results.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
        file.write("\n")

    csv_path = output_dir / "evaluation_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for method, metrics in methods.items():
            writer.writerow(
                {
                    "video": str(source_video),
                    "topic": topic,
                    "method": method,
                    "semantic_relevance": metrics["semantic_relevance"],
                    "content_coverage": metrics["content_coverage"],
                    "redundancy": metrics["redundancy"],
                }
            )

    print(f"Evaluation JSON: {json_path}")
    print(f"Evaluation CSV: {csv_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the three AI Video Repurposer research methods."
    )
    parser.add_argument("--video", required=True, type=Path, help="Source video.")
    parser.add_argument("--topic", required=True, help="Evaluation topic.")
    parser.add_argument(
        "--fixed-window",
        required=True,
        type=Path,
        help="Fixed-window baseline output directory.",
    )
    parser.add_argument(
        "--semantic-only",
        required=True,
        type=Path,
        help="Semantic-only baseline output directory.",
    )
    parser.add_argument(
        "--proposed",
        required=True,
        type=Path,
        help="Proposed pipeline clips directory (normally backend/clips).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory for evaluation_results.json and .csv.",
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.35,
        help=(
            "Minimum topic/source-segment cosine similarity for coverage "
            "(default: 0.35)."
        ),
    )
    parser.add_argument(
        "--whisper-model",
        default="tiny",
        help="Whisper model name used for source and clip transcripts.",
    )
    args = parser.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"Error: Source video does not exist: {args.video}")
    if not args.topic.strip():
        raise SystemExit("Error: Topic must not be empty.")
    if not 0 <= args.relevance_threshold <= 1:
        raise SystemExit("Error: --relevance-threshold must be from 0 to 1.")

    for directory in (
        args.fixed_window,
        args.semantic_only,
        args.proposed,
    ):
        if not directory.is_dir():
            raise SystemExit(f"Error: Method output directory does not exist: {directory}")

    evaluate(
        args.video,
        args.topic.strip(),
        {
            "fixed_window": args.fixed_window,
            "semantic_only": args.semantic_only,
            "proposed": args.proposed,
        },
        args.output,
        threshold=args.relevance_threshold,
        whisper_model_name=args.whisper_model,
    )


if __name__ == "__main__":
    main()
