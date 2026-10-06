import json
from pathlib import Path
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

BASE_DIR = Path(__file__).resolve().parent

VIDEO_FILE = BASE_DIR / "final_pipeline.mp4"
OUTPUT_FILE = BASE_DIR / "scene_segments.json"

def detect_scenes(video_path=VIDEO_FILE):
    video = open_video(str(video_path))

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))

    scene_manager.detect_scenes(video)

    scenes = scene_manager.get_scene_list()

    results = []

    for index, (start, end) in enumerate(scenes, 1):
        results.append({
            "scene": index,
            "start": round(start.get_seconds(), 2),
            "end": round(end.get_seconds(), 2),
            "duration": round(
                end.get_seconds() - start.get_seconds(), 2
            )
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\n=== SCENE SEGMENTATION ===")
    print(f"Scenes detected: {len(results)}")
    print(f"Video analyzed: {video_path}")

    for scene in results:
        print(
            f"Scene {scene['scene']}: "
            f"{scene['start']}s - {scene['end']}s "
            f"({scene['duration']}s)"
        )

    return results

if __name__ == "__main__":
    detect_scenes()
