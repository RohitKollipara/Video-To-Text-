import whisper
import json
from moviepy import VideoFileClip
from pathlib import Path


def extract_audio(video_path: str, audio_output: str = "extracted_audio.wav") -> str | None:
    """Extract audio from video file using MoviePy."""
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_output)
        video.close()
        print(f"Audio extracted to: {audio_output}")
        return audio_output
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def transcribe_video(video_file: str, model_size: str = "base",
                     output_dir: str = "transcription_output") -> dict | None:
    """
    Extract audio from video and transcribe it using Whisper.
    Saves plain text, timestamped transcript, and JSON to output directory.

    Args:
        video_file (str): Path to the video file
        model_size (str): Whisper model size - "tiny", "base", "small", "medium", "large"
        output_dir (str): Directory to save transcription files

    Returns:
        dict: Contains plain text, segments with timestamps
    """
    try:
        video_path = Path(video_file)
        if not video_path.exists():
            print(f"Error: File not found: {video_path}")
            return None

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        # Step 1: Extract audio from video
        print("Extracting audio from video...")
        audio_path = output / "extracted_audio.wav"
        extracted = extract_audio(str(video_path), str(audio_path))
        if not extracted:
            return None

        # Step 2: Transcribe with Whisper
        print(f"Loading Whisper model ({model_size})...")
        model = whisper.load_model(model_size)

        print("Transcribing... this may take a moment.")
        result = model.transcribe(
            str(audio_path),
            language="en",
            fp16=False,
            verbose=False,
            word_timestamps=False
        )

        # Step 3: Build structured transcription
        segments = []
        for seg in result["segments"]:
            segments.append({
                "start": round(seg["start"], 2),
                "end":   round(seg["end"], 2),
                "text":  seg["text"].strip()
            })

        plain_text = " ".join(s["text"] for s in segments)

        # Step 4: Save plain text
        plain_path = output / "transcription_plain.txt"
        plain_path.write_text(plain_text, encoding="utf-8")
        print(f"Plain text saved to: {plain_path}")

        # Step 5: Save timestamped transcript (readable)
        timestamped_path = output / "transcription_timestamped.txt"
        with timestamped_path.open("w", encoding="utf-8") as f:
            for seg in segments:
                start = format_time(seg["start"])
                end   = format_time(seg["end"])
                f.write(f"[{start} --> {end}]  {seg['text']}\n")
        print(f"Timestamped transcript saved to: {timestamped_path}")

        # Step 6: Save full JSON (for Gen AI pipeline use)
        json_path = output / "transcription_full.json"
        json_path.write_text(json.dumps(segments, indent=2), encoding="utf-8")
        print(f"Full JSON saved to: {json_path}")

        return {
            "plain_text": plain_text,
            "segments": segments
        }

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
    video_file ="D:\\Hackathon\\videos\cricket144.mp4"
    transcribe_video(video_file, model_size="base")
# ```

# **This produces 3 output files in `transcription_output/`:**

# - `transcription_plain.txt` — clean text for Gen AI input
# - `transcription_timestamped.txt` — readable segments like:
# ```
#   [00:00:03 --> 00:00:07]  The batsman takes his stance at the crease.
#   [00:00:07 --> 00:00:11]  He drives it through the covers for four!
