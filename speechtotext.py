import whisper
import noisereduce as nr
import soundfile as sf
import numpy as np
import json
from moviepy import VideoFileClip
from pathlib import Path


def clear_txt_files(output_dir: Path) -> None:
    """Delete only .txt files from previous run, keep .json files."""
    for txt_file in output_dir.glob("*.txt"):
        txt_file.unlink()
        print(f"Cleared: {txt_file.name}")


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


def denoise_audio(audio_path: str) -> tuple[np.ndarray, int] | None:
    """
    Denoise audio using noisereduce.
    Uses the first 0.5 seconds as a noise profile sample.
    """
    try:
        audio_data, sample_rate = sf.read(audio_path)

        if audio_data.ndim == 2:
            audio_data = np.mean(audio_data, axis=1)

        print("Denoising audio...")
        noise_sample_end = int(sample_rate * 0.5)
        noise_sample = audio_data[:noise_sample_end]

        denoised = nr.reduce_noise(
            y=audio_data,
            sr=sample_rate,
            y_noise=noise_sample,
            prop_decrease=0,
            stationary=False
        )

        print("Denoising complete.")
        return denoised, sample_rate

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def transcribe_video(video_file: str, model_size: str = "base",
                     output_dir: str = "transcription_output") -> dict | None:
    """
    Extract audio from video, denoise it, and transcribe using Whisper.
    Saves plain text, timestamped transcript, and JSON to output directory.

    Args:
        video_file (str): Path to the video file
        model_size (str): Whisper model size - "tiny", "base", "small", "medium", "large"
        output_dir (str): Directory to save transcription files

    Returns:
        dict: Contains plain text and segments with timestamps
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

        # Step 1: Extract audio
        print("Extracting audio from video...")
        audio_path = output / "extracted_audio.wav"
        extracted = extract_audio(str(video_path), str(audio_path))
        if not extracted:
            return None

        # Step 2: Denoise audio
        denoised_result = denoise_audio(str(audio_path))
        if denoised_result is None:
            print("Denoising failed, using original audio...")
            denoised_path = audio_path
        else:
            denoised_audio, sample_rate = denoised_result
            denoised_path = output / "denoised_audio.wav"
            sf.write(str(denoised_path), denoised_audio, sample_rate)
            print(f"Denoised audio saved to: {denoised_path}")

        # Step 3: Transcribe with Whisper
        print(f"Loading Whisper model ({model_size})...")
        model = whisper.load_model(model_size)

        print("Transcribing... this may take a moment.")
        result = model.transcribe(
            str(denoised_path),
            language="en",
            fp16=False,
            verbose=False,
            word_timestamps=False
        )

        # Step 4: Build structured transcription
        segments = []
        for seg in result["segments"]:
            segments.append({
                "start": round(seg["start"], 2),
                "end":   round(seg["end"],   2),
                "text":  seg["text"].strip()
            })

        plain_text = " ".join(s["text"] for s in segments)

        # Step 5: Save plain text .txt
        plain_path = output / "transcription_plain.txt"
        plain_path.write_text(plain_text, encoding="utf-8")
        print(f"Plain text saved to:          {plain_path}")

        # Step 6: Save timestamped .txt
        timestamped_path = output / "transcription_timestamped.txt"
        with timestamped_path.open("w", encoding="utf-8") as f:
            for seg in segments:
                start = format_time(seg["start"])
                end   = format_time(seg["end"])
                f.write(f"[{start} --> {end}]  {seg['text']}\n")
        print(f"Timestamped transcript saved: {timestamped_path}")

        # Step 7: Save .json (kept across runs)
        json_path = output / "transcription_full.json"
        json_path.write_text(json.dumps(segments, indent=2), encoding="utf-8")
        print(f"JSON saved to:                {json_path}")

        return {
            "plain_text": plain_text,
            "segments":   segments
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
    video_file = "D:\\Hackathon\\videos\\cricket480P.mp4"
    transcribe_video(video_file, model_size="base")
