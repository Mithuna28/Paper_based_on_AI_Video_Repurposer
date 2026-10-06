from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load MiniLM model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load transcript
with open("transcript_english/dsa_audio.json", "r", encoding="utf-8") as file:
    transcript = json.load(file)

# User topic
query = input("Enter the topic you want to find: ")

# Original Whisper segments
segments = transcript["segments"]

# Combine nearby transcript segments
chunks = []

chunk_text = ""
chunk_start = None
chunk_end = None

for segment in segments:

    if chunk_start is None:
        chunk_start = segment["start"]

    chunk_text += " " + segment["text"].strip()
    chunk_end = segment["end"]

    # Create a chunk approximately every 10 seconds
    if chunk_end - chunk_start >= 10:
        chunks.append({
            "start": chunk_start,
            "end": chunk_end,
            "text": chunk_text.strip()
        })

        chunk_text = ""
        chunk_start = None
        chunk_end = None

# Add remaining text
if chunk_text:
    chunks.append({
        "start": chunk_start,
        "end": chunk_end,
        "text": chunk_text.strip()
    })

# Extract chunk text
texts = [chunk["text"] for chunk in chunks]

# Convert query and transcript chunks into embeddings
query_embedding = model.encode([query])
text_embeddings = model.encode(texts)

# Calculate similarity
scores = cosine_similarity(query_embedding, text_embeddings)[0]

# Store results
results = []

for i, score in enumerate(scores):

    results.append({
        "start": chunks[i]["start"],
        "end": chunks[i]["end"],
        "text": chunks[i]["text"],
        "score": float(score)
    })

# Sort by similarity
results.sort(key=lambda x: x["score"], reverse=True)

# Select relevant segments
threshold = 0.70

selected_segments = [
    result for result in results
    if result["score"] >= threshold
]

# Sort selected clips by original video timeline
selected_segments.sort(key=lambda x: x["start"])

print("\nSelected relevant segments:\n")

for result in selected_segments:

    print(
        f"[{result['start']:.2f}s - {result['end']:.2f}s] "
        f"Score: {result['score']:.3f}"
    )

    print(result["text"])
    print()

# Save selected segments
with open("selected_segments.json", "w", encoding="utf-8") as file:
    json.dump(selected_segments, file, indent=4)

print("\nSelected segments saved to selected_segments.json")