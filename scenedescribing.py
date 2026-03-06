import cv2
import base64
import json
import time
from pathlib import Path
from groq import Groq


def clear_txt_files(output_dir: Path) -> None:
    """Delete only .txt files from previous run, keep .json files."""
    for txt_file in output_dir.glob("*.txt"):
        txt_file.unlink()
        print(f"Cleared: {txt_file.name}")


def extract_frames(video_path: str, interval_seconds: int = 4) -> list[dict]:
    """Extract one frame every `interval_seconds` from the video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    frame_interval = int(fps * interval_seconds)
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            timestamp = round(frame_index / fps, 2)
            _, buffer = cv2.imencode(".jpg", frame)
            frames.append({
                "timestamp": timestamp,
                "bytes": buffer.tobytes()
            })

        frame_index += 1

    cap.release()
    print(f"Extracted {len(frames)} frames (every {interval_seconds}s)")
    return frames

def describe_frames(frames: list[dict], api_key: str) -> list[dict]:
    """Send each frame to Groq Vision and extract rich semantic context."""
    client = Groq(api_key=api_key)

    prompt = """You are a video analyst helping recreate a video using Gen AI.
    Analyze this frame and return a JSON object with exactly these fields:

    {
    "action": "What is physically happening in the scene — any activity, movement, event, or interaction taking place.",
    "camera": "Camera angle and shot style — wide shot, close-up, medium shot, aerial, point-of-view, slow-motion, static, panning, etc.",
    "emotion": "Emotions and body language of any people present — expressions, posture, energy level. If no people, describe the mood of the scene.",
    "atmosphere": "Overall environment and ambiance — indoor/outdoor, crowd or no crowd, time of day, lighting, energy level, setting type.",
    "context": "One detailed sentence combining all the above into a complete scene description for video generation."
    }

    Return ONLY the JSON object, no extra text."""

    descriptions = []
    total = len(frames)

    for i, frame in enumerate(frames):
        timestamp = frame["timestamp"]
        print(f"Analyzing frame {i + 1}/{total} at {format_time(timestamp)}...")

        try:
            image_b64 = base64.b64encode(frame["bytes"]).decode("utf-8")

            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=400
            )

            raw = response.choices[0].message.content.strip()

            # Clean markdown code blocks if present
            raw = raw.replace("```json", "").replace("```", "").strip()

            # Extract only the JSON object even if there's extra text around it
            start_idx = raw.find("{")
            end_idx   = raw.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                raw = raw[start_idx:end_idx]

            try:
                semantic = json.loads(raw)
            except json.JSONDecodeError:
                semantic = {
                    "action":     "unknown",
                    "camera":     "unknown",
                    "emotion":    "unknown",
                    "atmosphere": "unknown",
                    "context":    raw
                }

        except Exception as e:
            semantic = {
                "action":     "error",
                "camera":     "error",
                "emotion":    "error",
                "atmosphere": "error",
                "context":    f"[Error: {e}]"
            }

        time.sleep(1)

        descriptions.append({
            "timestamp": timestamp,
            "semantic":  semantic
        })

    return descriptions

def describe_video(video_file: str, api_key: str, interval_seconds: int = 2,
                   output_dir: str = "transcription_output") -> list[dict] | None:
    """
    Extract frames and generate rich semantic scene descriptions using Groq.

    Args:
        video_file (str): Path to the video file
        api_key (str): Groq API key
        interval_seconds (int): How often to sample frames (default: every 4 seconds)
        output_dir (str): Directory to save output files

    Returns:
        list[dict]: List of timestamped semantic descriptions
    """
    try:
        video_path = Path(video_file)
        if not video_path.exists():
            print(f"Error: File not found: {video_path}")
            return None

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        # Clear only .txt files from previous run
        print("Clearing old .txt files...")
        clear_txt_files(output)

        # Step 1: Extract frames
        print("Extracting frames from video...")
        frames = extract_frames(str(video_path), interval_seconds)
        if not frames:
            print("No frames extracted.")
            return None

        # Step 2: Describe each frame
        print("Sending frames to Groq for semantic analysis...")
        descriptions = describe_frames(frames, api_key)

        # Step 3: Save readable .txt
        readable_path = output / "scene_descriptions.txt"
        with readable_path.open("w", encoding="utf-8") as f:
            for d in descriptions:
                time_str = format_time(d["timestamp"])
                s = d["semantic"]
                f.write(f"[{time_str}]\n")
                f.write(f"  ACTION:     {s.get('action',     'N/A')}\n")
                f.write(f"  CAMERA:     {s.get('camera',     'N/A')}\n")
                f.write(f"  EMOTION:    {s.get('emotion',    'N/A')}\n")
                f.write(f"  ATMOSPHERE: {s.get('atmosphere', 'N/A')}\n")
                f.write(f"  CONTEXT:    {s.get('context',    'N/A')}\n\n")
        print(f"Scene descriptions saved to: {readable_path}")

        # Step 4: Save .json (kept across runs)
        json_path = output / "scene_descriptions.json"
        json_path.write_text(json.dumps(descriptions, indent=2), encoding="utf-8")
        print(f"JSON saved to:               {json_path}")

        return descriptions

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"


if __name__ == "__main__":
    video_file = "D:\\Hackathon\\videos\\cricket480P.mp4"
    api_key = "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"
    describe_video(video_file, api_key, interval_seconds=2)
