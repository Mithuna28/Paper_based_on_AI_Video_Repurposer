import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "optimized_segments.json"
OUTPUT_FILE = BASE_DIR / "unique_segments.json"

SIMILARITY_THRESHOLD = 0.80


def remove_duplicate_segments(segments, model):
    if not segments:
        return []

    texts = [segment["text"] for segment in segments]
    embeddings = model.encode(texts)

    unique_segments = []
    unique_indices = []

    for index, segment in enumerate(segments):
        is_duplicate = False

        for existing_index, existing in zip(unique_indices, unique_segments):
            similarity = cosine_similarity(
                [embeddings[index]],
                [embeddings[existing_index]]
            )[0][0]

            if similarity >= SIMILARITY_THRESHOLD:
                is_duplicate = True
                print(
                    f"Removed redundant clip: "
                    f"{segment['start']:.2f}s - {segment['end']:.2f}s "
                    f"(similarity={similarity:.3f})"
                )
                break

        if not is_duplicate:
            unique_segments.append(segment)
            unique_indices.append(index)

    return unique_segments


def remove_duplicates():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        segments = json.load(file)

    if not segments:
        print("No segments found.")
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")
    unique_segments = remove_duplicate_segments(segments, model)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            unique_segments,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n=== DUPLICATE REMOVAL ===")
    print(f"Original clips : {len(segments)}")
    print(f"Unique clips   : {len(unique_segments)}")
    print(f"Removed clips  : {len(segments) - len(unique_segments)}")
    print(f"Output         : {OUTPUT_FILE.name}")


if __name__ == "__main__":
    remove_duplicates()
