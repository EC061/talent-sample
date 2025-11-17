#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate 5-second segment descriptions for YouTube videos.

For each video listed in `materials/video/video.csv`:
  - Download the video with `yt-dlp`
  - Segment the video into 5-second chunks with `ffmpeg`
  - (If using OpenAI platform) transcribe audio for each chunk
  - Capture a representative frame for each chunk
  - Call the vision-language model once per chunk with:
      - the image (visual content)
      - the transcript text (voice content, if available)
  - Store the result in `materials/processed_materials.db` in a `video` table
    with:
      - video_category
      - video_name
      - video_link
      - segment_index
      - start_second
      - end_second
      - voice_content
      - visual_content
      - description
      - needed (0/1, where 1 means learning-relevant)
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import ffmpeg

from pdf_description_gen import MaterialsDescriptionGenerator
from config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_DB_PATH = PROJECT_ROOT / "materials" / "processed_materials.db"
DEFAULT_CSV_PATH = PROJECT_ROOT / "materials" / "video" / "video.csv"
DEFAULT_DOWNLOAD_DIR = PROJECT_ROOT / "materials" / "video" / "downloads"


@dataclass
class VideoMeta:
    category: str
    name: str
    link: str


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    """Run a subprocess command, raising on error."""
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd) if cwd is not None else None,
        )
    except subprocess.CalledProcessError as e:
        stdout = e.stdout.decode("utf-8", errors="ignore") if e.stdout else ""
        stderr = e.stderr.decode("utf-8", errors="ignore") if e.stderr else ""
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        ) from e


def safe_filename(name: str) -> str:
    """Create a filesystem-safe base name."""
    keep = "-_.() "
    cleaned = "".join(c for c in name if c.isalnum() or c in keep).strip()
    return cleaned.replace(" ", "_") or "video"


def read_video_csv(csv_path: Path) -> List[VideoMeta]:
    """Read video list from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Video CSV not found at {csv_path}")

    videos: List[VideoMeta] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        # Skip header
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 3:
                continue
            category, name, link = row[:3]
            category = (category or "").strip()
            name = (name or "").strip()
            link = (link or "").strip()
            if not link:
                continue
            videos.append(VideoMeta(category=category, name=name, link=link))
    return videos


def download_video(meta: VideoMeta, download_dir: Path) -> Path:
    """
    Download a YouTube video using yt-dlp.

    Returns:
        Path to the downloaded video file.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    base = safe_filename(meta.name or meta.category or "video")
    # Let yt-dlp choose extension, but constrain to mp4 container for ffmpeg convenience.
    output_template = f"{base}.%(ext)s"

    cmd = [
        "yt-dlp",
        "-f",
        "mp4",
        "-o",
        output_template,
        meta.link,
    ]

    run_cmd(cmd, cwd=download_dir)

    # Find the downloaded file
    candidates = list(download_dir.glob(f"{base}.*"))
    if not candidates:
        raise FileNotFoundError(f"yt-dlp completed but no file found matching {base}.* in {download_dir}")

    # Prefer mp4 if multiple
    mp4s = [p for p in candidates if p.suffix.lower() == ".mp4"]
    return mp4s[0] if mp4s else candidates[0]


def get_video_duration(video_path: Path) -> float:
    try:
        probe = ffmpeg.probe(str(video_path))
        fmt = probe.get("format", {})
        duration_str = fmt.get("duration")
        if duration_str is None:
            raise RuntimeError(f"No duration found in ffprobe output for {video_path}")
    except ffmpeg.Error as e:
        raise RuntimeError(f"ffmpeg.probe failed for {video_path}: {e}") from e

    try:
        return float(duration_str)
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse duration '{duration_str}' from ffprobe output for {video_path}"
        ) from exc


def extract_segment(
    video_path: Path,
    segments_dir: Path,
    base_name: str,
    segment_index: int,
    start: float,
    end: float,
) -> Tuple[Path, Path, Path]:
    """
    Extract a 5-second video segment, its audio track, and a representative frame.

    Returns:
        (segment_video_path, segment_audio_path, segment_frame_path)
    """
    segments_dir.mkdir(parents=True, exist_ok=True)
    seg_tag = f"{base_name}_seg_{segment_index:04d}"
    seg_video = segments_dir / f"{seg_tag}.mp4"
    seg_audio = segments_dir / f"{seg_tag}.wav"
    seg_frame = segments_dir / f"{seg_tag}.jpg"

    duration = max(0.0, end - start)
    if duration <= 0.2:
        raise ValueError(f"Segment duration too short ({duration:.2f}s) for index {segment_index}")

    # 1) Cut video segment
    (
        ffmpeg
        .input(str(video_path), ss=start, t=duration)
        .output(str(seg_video), c="copy", loglevel="error")
        .overwrite_output()
        .run()
    )

    # 2) Extract mono 16kHz wav audio
    (
        ffmpeg
        .input(str(seg_video))
        .output(
            str(seg_audio),
            vn=None,
            acodec="pcm_s16le",
            ar="16000",
            ac=1,
            loglevel="error",
        )
        .overwrite_output()
        .run()
    )

    # 3) Capture one frame (first frame of segment)
    (
        ffmpeg
        .input(str(seg_video))
        .output(str(seg_frame), vframes=1, loglevel="error")
        .overwrite_output()
        .run()
    )

    return seg_video, seg_audio, seg_frame


def transcribe_audio_if_available(
    generator: MaterialsDescriptionGenerator,
    audio_path: Path,
) -> str:
    """
    Transcribe audio using the OpenAI audio transcription API.
    """
    # Choose a reasonable default transcription model; can be overridden via config.yml.
    cfg = load_config()
    api_cfg = cfg.get("api", {}).get("openai", {})
    audio_model = api_cfg.get("audio_transcription_model", "gpt-4o-mini-transcribe")

    try:
        with audio_path.open("rb") as f:
            resp = generator.client.audio.transcriptions.create(
                model=audio_model,
                file=f,
            )
        # Both whisper-1 and gpt-4o-mini-transcribe expose `.text`
        return getattr(resp, "text", "") or ""
    except Exception as e:
        print(f"Warning: audio transcription failed for {audio_path}: {e}")
        return ""


def ensure_video_table(db_path: Path) -> None:
    """Create the `video` table if it does not already exist."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS video (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_category TEXT,
                video_name TEXT,
                video_link TEXT,
                segment_index INTEGER,
                start_second REAL,
                end_second REAL,
                voice_content TEXT,
                visual_content TEXT,
                description TEXT,
                needed INTEGER
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def describe_segment(
    generator: MaterialsDescriptionGenerator,
    frame_path: Path,
    transcript: str,
) -> Dict[str, Any]:
    """
    Call the VLM to describe a single 5-second segment using both visual and voice content.

    Returns:
        Parsed JSON with keys: needed (bool), voice_content, visual_content, description
    """
    transcript = (transcript or "").strip()
    if not transcript:
        transcript_for_prompt = "NO SPEECH DETECTED"
    else:
        transcript_for_prompt = transcript

    # Structured output schema
    schema: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "needed": {
                "type": "boolean",
                "description": (
                    "true if this segment teaches a substantive educational concept, explanation, "
                    "worked example, or problem-solving step; false if it is an advertisement, "
                    "intro/outro bumper, idle time, jokes, or other non-learning content."
                ),
            },
            "voice_content": {
                "type": "string",
                "description": "1–2 sentences summarizing what is said in the audio, or 'none' if no speech.",
            },
            "visual_content": {
                "type": "string",
                "description": "1–2 sentences describing what is visually shown on screen.",
            },
            "description": {
                "type": "string",
                "description": "1–2 sentences combining both audio and visuals to describe what the student learns.",
            },
        },
        "required": ["needed", "voice_content", "visual_content", "description"],
    }

    prompt = (
        "You are analyzing a single 5-second segment from an educational physics video.\n\n"
        "Use BOTH the provided video frame (image) and the transcript text below to understand "
        "what is happening in this short segment.\n\n"
        "Transcript (may be empty if there is no speech):\n"
        f"\"\"\"{transcript_for_prompt}\"\"\"\n\n"
        "Return ONLY a JSON object with fields exactly:\n"
        "  - needed (boolean)\n"
        "  - voice_content (string)\n"
        "  - visual_content (string)\n"
        "  - description (string)\n\n"
        "Rules for needed:\n"
        "- needed = true only if the segment clearly teaches or explains a physics concept, shows a worked example, "
        "  or advances a problem-solving explanation.\n"
        "- needed = false for advertisements, sponsor messages, intros/outros, music-only segments, "
        "  waiting/idle time, jokes, or other non-learning material.\n\n"
        "Output strict JSON only with no extra text."
    )

    result = generator.generate_description(
        str(frame_path),
        prompt,
        use_url=False,
        return_metrics=False,
        guided_json=schema,
    )

    if isinstance(result, dict) and "content" in result:
        content = result["content"]
    else:
        content = result

    if not isinstance(content, str):
        raise ValueError(f"Unexpected model response type: {type(content)}")

    try:
        parsed = json.loads(content)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON from model response: {e}; content: {content}") from e

    # Basic normalization
    needed_val = parsed.get("needed", None)
    if isinstance(needed_val, bool):
        parsed["needed"] = needed_val
    elif isinstance(needed_val, (int, float)):
        parsed["needed"] = bool(needed_val)

    return parsed


def process_video(
    generator: MaterialsDescriptionGenerator,
    meta: VideoMeta,
    db_path: Path,
    download_root: Path,
) -> None:
    """Process a single video into 5-second segments and store descriptions in the DB."""
    print(f"\n=== Processing video: {meta.name} ({meta.link}) ===")
    download_dir = download_root
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        video_path = download_video(meta, download_dir)
    except Exception as e:
        print(f"✗ Failed to download video '{meta.name}': {e}")
        return

    print(f"Downloaded to: {video_path}")

    try:
        duration = get_video_duration(video_path)
    except Exception as e:
        print(f"✗ Failed to get duration for '{video_path}': {e}")
        return

    print(f"Duration: {duration:.2f} seconds")

    # Prepare DB
    ensure_video_table(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    base_name = safe_filename(meta.name or meta.category or "video")
    segments_dir = download_root / f"{base_name}_segments"

    num_segments = max(1, int(math.ceil(duration / 5.0)))
    print(f"Segmenting into {num_segments} chunks of 5 seconds (last may be shorter)")

    for idx in range(num_segments):
        start = idx * 5.0
        end = min((idx + 1) * 5.0, duration)
        if end - start <= 0.2:
            # Skip extremely short tail segment
            continue

        print(f"  - Segment {idx}: {start:.2f}s -> {end:.2f}s")

        try:
            seg_video, seg_audio, seg_frame = extract_segment(
                video_path=video_path,
                segments_dir=segments_dir,
                base_name=base_name,
                segment_index=idx,
                start=start,
                end=end,
            )
        except Exception as e:
            print(f"    ✗ Failed to extract segment {idx}: {e}")
            continue

        transcript = transcribe_audio_if_available(generator, seg_audio)

        try:
            desc = describe_segment(generator, seg_frame, transcript)
        except Exception as e:
            print(f"    ✗ Model error for segment {idx}: {e}")
            continue

        needed_bool = bool(desc.get("needed", False))
        needed_int = 1 if needed_bool else 0
        voice_text = (desc.get("voice_content") or "").strip()
        visual_text = (desc.get("visual_content") or "").strip()
        description = (desc.get("description") or "").strip()

        cur.execute(
            """
            INSERT INTO video (
                video_category,
                video_name,
                video_link,
                segment_index,
                start_second,
                end_second,
                voice_content,
                visual_content,
                description,
                needed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meta.category,
                meta.name,
                meta.link,
                idx,
                float(start),
                float(end),
                voice_text,
                visual_text,
                description,
                needed_int,
            ),
        )
        conn.commit()

        label = "needed" if needed_bool else "unneeded"
        print(f"    ✓ Stored segment {idx} ({label})")

    conn.close()
    print(f"✓ Finished processing video: {meta.name}")


def main(
    db_path: Optional[str] = None,
    csv_path: Optional[str] = None,
) -> None:
    """Entry point: process all videos listed in the CSV."""
    # Initialize VLM generator using same configuration mechanism as pdf_description_gen
    generator = MaterialsDescriptionGenerator()

    db_path_obj = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    csv_path_obj = Path(csv_path) if csv_path is not None else DEFAULT_CSV_PATH

    print(f"Using DB: {db_path_obj}")
    print(f"Using video CSV: {csv_path_obj}")

    videos = read_video_csv(csv_path_obj)
    if not videos:
        print("No videos found in CSV. Nothing to do.")
        return

    print(f"Found {len(videos)} video(s) in CSV.")

    for meta in videos:
        process_video(
            generator=generator,
            meta=meta,
            db_path=db_path_obj,
            download_root=DEFAULT_DOWNLOAD_DIR,
        )


if __name__ == "__main__":
    # Allow optional CLI override: python video_description_gen.py [db_path] [csv_path]
    db_arg = sys.argv[1] if len(sys.argv) > 1 else None
    csv_arg = sys.argv[2] if len(sys.argv) > 2 else None
    main(db_path=db_arg, csv_path=csv_arg)


