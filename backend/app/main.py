import base64
import logging
import math
import os
import subprocess
import sys
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.database import jobs_collection
from title_service import generate_title, generate_hashtags
from video_validator import validate_video_file


# =========================
# APP
# =========================

app = FastAPI(title="AI Video Repurposer")
logger = logging.getLogger(__name__)


# =========================
# APP STATE
# =========================

app.state.result_ready = False
app.state.result_video = None
app.state.latest_job_id = None

app.state.progress = {
    "stage": "idle",
    "progress": 0
}


def update_progress(stage, progress):
    app.state.progress = {
        "stage": stage,
        "progress": progress
    }

    print(f"[PROGRESS] {progress}% - {stage}")


# =========================
# EMOJI MODELS
# =========================

class EmojiAsset(BaseModel):
    id: str
    image_base64: str


class EmojiOverlay(BaseModel):
    emoji: str
    start: float
    end: float
    caption: str
    asset_id: str


class EmojiProcessingRequest(BaseModel):
    overlays: list[EmojiOverlay]
    assets: list[EmojiAsset]


# =========================
# EMOJI FILTER
# =========================

def build_emoji_filter(overlays):
    filters = ["[0:v]setpts=PTS-STARTPTS[base0]"]
    previous_label = "base0"

    for index, overlay in enumerate(overlays):
        start = f"{overlay.start:.3f}"
        end = f"{overlay.end:.3f}"

        duration = overlay.end - overlay.start

        fade_in = min(0.25, duration / 2)
        fade_out = min(0.3, duration / 2)

        fade_out_start = overlay.end - fade_out

        input_label = f"emoji{index}"
        output_label = f"base{index + 1}"

        scale_progress = f"(t-{start})/{fade_in:.3f}"

        filters.append(
            f"[{index + 1}:v]format=rgba,"
            f"setpts=PTS-STARTPTS+{start}/TB,"
            f"scale=w='iw*(0.7+0.3*min(1,max(0,{scale_progress})))':"
            f"h='ih*(0.7+0.3*min(1,max(0,{scale_progress})))':eval=frame,"
            f"fade=t=in:st={start}:d={fade_in:.3f}:alpha=1,"
            f"fade=t=out:st={fade_out_start:.3f}:d={fade_out:.3f}:alpha=1"
            f"[{input_label}]"
        )

        filters.append(
            f"[{previous_label}][{input_label}]"
            f"overlay=x=(W-w)/2:y=H-h-280:"
            f"enable='between(t,{start},{end})'[{output_label}]"
        )

        previous_label = output_label

    return ";".join(filters), previous_label


# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# DIRECTORIES
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# =========================
# HOME
# =========================

@app.get("/")
def home():
    return {
        "message": "AI Video Repurposer API is running"
    }


# =========================
# HEALTH CHECK
# =========================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "message": "AI Video Repurposer backend is running"
    }


# =========================
# PROGRESS
# =========================

@app.get("/api/video/progress")
def get_progress():
    return app.state.progress


# =========================
# VIDEO PROCESSING
# =========================

@app.post("/api/video/process")
async def process_video(
    file: UploadFile = File(...),
    topic: str = Form(...)
):
    app.state.result_ready = False
    app.state.result_video = None
    app.state.latest_job_id = None

    update_progress("Starting", 0)

    try:

        # -------------------------
        # Create Job ID
        # -------------------------

        job_id = str(uuid.uuid4())
        app.state.latest_job_id = job_id

        print("=" * 60)
        print(f"[API] Job ID: {job_id}")
        print(f"[API] Target Topic: {topic}")
        print(f"[API] Uploaded File: {file.filename}")
        print("=" * 60)

        # -------------------------
        # Save Uploaded Video
        # -------------------------

        video_path = UPLOAD_DIR / file.filename

        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"[API] Video saved: {video_path}")

        update_progress("Video uploaded", 10)

        # -------------------------
        # Create MongoDB Job
        # -------------------------

        jobs_collection.insert_one({
            "job_id": job_id,
            "filename": file.filename,
            "topic": topic,
            "status": "processing",
            "progress": 0,
            "created_at": datetime.utcnow()
        })

        print(f"[API] Job stored in MongoDB: {job_id}")

        update_progress("Job created", 15)

        # -------------------------
        # Pipeline Path
        # -------------------------

        pipeline_path = BASE_DIR / "pipeline.py"

        if not pipeline_path.exists():
            raise HTTPException(
                status_code=500,
                detail="pipeline.py not found"
            )

        # -------------------------
        # Run AI Video Pipeline
        # -------------------------

        print("[API] Starting AI video processing...")

        update_progress("AI processing started", 20)

        result = subprocess.run(
            [
                sys.executable,
                str(pipeline_path),
                file.filename,
                topic
            ],
            capture_output=True,
            text=True
        )

        # -------------------------
        # Pipeline Failed
        # -------------------------

        if result.returncode != 0:

            print("[API ERROR] Pipeline failed")

            print("STDOUT:")
            print(result.stdout)

            print("STDERR:")
            print(result.stderr)

            jobs_collection.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "status": "failed",
                        "error": result.stderr or result.stdout
                    }
                }
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "Video processing failed:\n"
                    + (result.stderr or result.stdout)
                )
            )

        # -------------------------
        # Show Pipeline Output
        # -------------------------

        print("[API] Pipeline completed successfully")

        update_progress("AI processing completed", 80)

        if result.stdout:
            print(result.stdout)

        # -------------------------
        # Check Final Video
        # -------------------------

        final_video = BASE_DIR / "final_pipeline.mp4"

        if not final_video.exists():

            jobs_collection.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "status": "failed",
                        "error": "Final video was not created"
                    }
                }
            )

            print("[API ERROR] Final video was not created")

            update_progress("Final video missing", 0)

            raise HTTPException(
                status_code=500,
                detail="Final video was not created"
            )

        # -------------------------
        # Validate Final Video
        # -------------------------

        print("[API] Validating final video...")

        try:
            validate_video_file(final_video)

        except ValueError as error:

            jobs_collection.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "status": "failed",
                        "error": str(error)
                    }
                }
            )

            print(f"[API ERROR] Validation failed: {error}")

            update_progress("Video validation failed", 0)

            raise HTTPException(
                status_code=409,
                detail=f"Final video failed media validation: {error}"
            ) from error

        update_progress("Video validated", 90)

        # -------------------------
        # Mark Result Ready
        # -------------------------

        app.state.result_ready = True
        app.state.result_video = final_video.name

        # -------------------------
        # Update MongoDB
        # -------------------------

        jobs_collection.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "completed",
                    "progress": 100,
                    "output_file": "final_pipeline.mp4",
                    "completed_at": datetime.utcnow()
                }
            }
        )

        print("=" * 60)
        print("[API] VIDEO PROCESSING COMPLETED")
        print(f"[API] Output: {final_video}")
        print(f"[API] MongoDB job completed: {job_id}")
        print("=" * 60)

        # -------------------------
        # Read Selected Captions
        # -------------------------

        captions_file = BASE_DIR / "selected_captions.srt"

        captions = ""

        try:
            captions = captions_file.read_text(
                encoding="utf-8"
            )

        except OSError as error:

            print(
                "[API WARNING] Could not read selected captions "
                f"for AI generation: {error}"
            )

        # -------------------------
        # AI TITLE + HASHTAGS
        # -------------------------

        update_progress("Generating AI title", 95)

        title = "AI title unavailable"
        hashtags = []

        # -------------------------
        # Generate AI Title
        # -------------------------

        try:

            title = generate_title(
                topic,
                captions
            )

            print(f"[AI TITLE] {title}")

        except Exception:

            logger.exception(
                "[AI TITLE ERROR] Generation failed for job %s",
                job_id
            )

        # -------------------------
        # Generate AI Hashtags
        # -------------------------

        try:

            hashtags = generate_hashtags(
                topic,
                captions
            )

            print(f"[AI HASHTAGS] {hashtags}")

        except Exception:

            logger.exception(
                "[AI HASHTAG ERROR] Generation failed for job %s",
                job_id
            )

        # -------------------------
        # Store AI Title
        # -------------------------

        try:

            jobs_collection.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "title": title
                    }
                }
            )

        except Exception:

            logger.exception(
                "Could not store generated title for job %s",
                job_id
            )

        # -------------------------
        # Processing Complete
        # -------------------------

        update_progress("Completed", 100)

        # -------------------------
        # Return Response
        # -------------------------

        return {
            "status": "success",
            "message": "Video processed and validated successfully",
            "job_id": job_id,
            "topic": topic,
            "video": "final_pipeline.mp4",
            "captions": captions,
            "title": title,
            "hashtags": hashtags
        }

    # -------------------------
    # HTTP Exception
    # -------------------------

    except HTTPException:
        raise

    # -------------------------
    # General Exception
    # -------------------------

    except Exception as e:

        print("=" * 60)
        print("[API EXCEPTION]")
        print(str(e))
        print("=" * 60)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================
# ADD EMOJIS
# =========================

@app.post("/api/video/add-emojis")
def add_emojis(request: EmojiProcessingRequest):

    if not app.state.result_ready:
        raise HTTPException(
            status_code=409,
            detail="No validated result is available. Process a video first."
        )

    final_video = BASE_DIR / "final_pipeline.mp4"

    if not final_video.is_file():
        raise HTTPException(
            status_code=404,
            detail="Final video not found"
        )

    app.state.result_video = final_video.name

    # -------------------------
    # No Emoji Overlay
    # -------------------------

    if not request.overlays:
        return {
            "status": "fallback",
            "video": final_video.name
        }

    try:

        # -------------------------
        # Validate Emoji Assets
        # -------------------------

        asset_data_by_id = {}

        for asset in request.assets:

            if not asset.id or asset.id in asset_data_by_id:
                raise ValueError(
                    "Emoji asset IDs must be non-empty and unique."
                )

            if len(asset.image_base64) > 2_000_000:
                raise ValueError(
                    "An emoji PNG exceeds the allowed size."
                )

            image_data = base64.b64decode(
                asset.image_base64,
                validate=True
            )

            if not image_data.startswith(
                b"\x89PNG\r\n\x1a\n"
            ):
                raise ValueError(
                    f"Emoji asset {asset.id} is not a PNG image."
                )

            asset_data_by_id[asset.id] = image_data

        # -------------------------
        # Validate Overlay Data
        # -------------------------

        for overlay in request.overlays:

            if (
                not math.isfinite(overlay.start)
                or not math.isfinite(overlay.end)
                or overlay.start < 0
                or overlay.end <= overlay.start
            ):
                raise ValueError(
                    "Emoji overlay timestamps are invalid."
                )

            if overlay.asset_id not in asset_data_by_id:
                raise ValueError(
                    f"Emoji overlay references missing asset "
                    f"{overlay.asset_id}."
                )

        # -------------------------
        # Temporary Rendering
        # -------------------------

        with tempfile.TemporaryDirectory(
            prefix="emoji_render_",
            dir=BASE_DIR
        ) as temporary_directory:

            temporary_path = Path(temporary_directory)

            asset_paths = {}

            for index, (
                asset_id,
                image_data
            ) in enumerate(asset_data_by_id.items()):

                asset_path = (
                    temporary_path /
                    f"emoji_{index}.png"
                )

                asset_path.write_bytes(image_data)

                asset_paths[asset_id] = asset_path

            output_path = (
                temporary_path /
                "rendered.mp4"
            )

            # -------------------------
            # Build FFmpeg Filter
            # -------------------------

            filter_graph, output_label = build_emoji_filter(
                request.overlays
            )

            command = [
                "ffmpeg",
                "-y",
                "-i",
                str(final_video)
            ]

            # -------------------------
            # Add Emoji Inputs
            # -------------------------

            for overlay in request.overlays:

                command.extend([
                    "-loop",
                    "1",
                    "-framerate",
                    "30",
                    "-i",
                    str(asset_paths[overlay.asset_id])
                ])

            # -------------------------
            # FFmpeg Output
            # -------------------------

            command.extend([
                "-filter_complex",
                filter_graph,

                "-map",
                f"[{output_label}]",

                "-map",
                "0:a?",

                "-c:v",
                "libx264",

                "-pix_fmt",
                "yuv420p",

                "-c:a",
                "copy",

                "-shortest",

                "-movflags",
                "+faststart",

                str(output_path)
            ])

            # -------------------------
            # Run FFmpeg
            # -------------------------

            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:

                logger.error(
                    "FFmpeg emoji overlay failed: %s",
                    result.stderr[-4000:]
                )

                raise RuntimeError(
                    "FFmpeg could not render the emoji overlays."
                )

            # -------------------------
            # Validate Emoji Video
            # -------------------------

            validate_video_file(output_path)

            emoji_video = (
                BASE_DIR /
                "final_pipeline_emoji.mp4"
            )

            os.replace(
                output_path,
                emoji_video
            )

        # -------------------------
        # Update MongoDB
        # -------------------------

        if app.state.latest_job_id:

            jobs_collection.update_one(
                {
                    "job_id":
                    app.state.latest_job_id
                },
                {
                    "$set": {
                        "output_file":
                            "final_pipeline_emoji.mp4",

                        "emoji_processing":
                            "completed",

                        "completed_at":
                            datetime.now(timezone.utc)
                    }
                }
            )

        # -------------------------
        # Update Result
        # -------------------------

        app.state.result_video = (
            "final_pipeline_emoji.mp4"
        )

        return {
            "status": "success",
            "video": "final_pipeline_emoji.mp4"
        }

    except (
        ValueError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError
    ) as error:

        logger.exception(
            "Emoji rendering failed; keeping the original video: %s",
            error
        )

        return {
            "status": "fallback",
            "video": "final_pipeline.mp4",
            "message": (
                "Emoji rendering failed; "
                "the original video is available."
            )
        }


# =========================
# GET FINAL VIDEO
# =========================

@app.get("/api/video/result")
def get_result_video():

    if not app.state.result_ready:

        raise HTTPException(
            status_code=409,
            detail=(
                "No validated result is available. "
                "Process a video first."
            )
        )

    # -------------------------
    # Allowed Result Files
    # -------------------------

    result_filename = app.state.result_video

    if result_filename not in (
        "final_pipeline.mp4",
        "final_pipeline_emoji.mp4"
    ):

        raise HTTPException(
            status_code=404,
            detail="Final video not found"
        )

    final_video = (
        BASE_DIR /
        result_filename
    )

    # -------------------------
    # Check File
    # -------------------------

    if not final_video.exists():

        raise HTTPException(
            status_code=404,
            detail="Final video not found"
        )

    # -------------------------
    # Validate File
    # -------------------------

    try:

        validate_video_file(
            final_video
        )

    except ValueError as error:

        raise HTTPException(
            status_code=409,
            detail=(
                f"Final video failed media validation: {error}"
            )
        ) from error

    # -------------------------
    # Cache Headers
    # -------------------------

    headers = {
        "Cache-Control":
            "no-cache, no-store, must-revalidate",

        "Pragma":
            "no-cache",

        "Expires":
            "0"
    }

    # -------------------------
    # Return Video
    # -------------------------

    return FileResponse(
        final_video,
        media_type="video/mp4",
        filename=result_filename,
        headers=headers
    )