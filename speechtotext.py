import whisper
from pathlib import Path


def recognize_speech(audio_file: str, model_size: str = "base") -> str | None:
    """
    Recognize speech from an audio file using OpenAI Whisper.

    Args:
        audio_file (str): Path to the audio file (.wav, .mp3, .m4a, .flac, etc.)
        model_size (str): Whisper model size - "tiny", "base", "small", "medium", "large"

    Returns:
        str: Recognized text from speech, or None if failed
    """
    try:
        audio_path = Path(audio_file)

        if not audio_path.exists():
            print(f"Error: File not found: {audio_path}")
            return None

        if not audio_path.is_file():
            print(f"Error: Path is not a file: {audio_path}")
            return None

        print(f"Loading audio from: {audio_path}")
        print("Processing speech recognition...")

        model = whisper.load_model(model_size)
        result = model.transcribe(str(audio_path), language="en", fp16=False)
        text = result["text"].strip()

        if text:
            print(f"Recognized text: {text}")
            return text
        else:
            print("No speech detected.")
            return None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    audio_file = "file.mp3"
    result = recognize_speech(audio_file)

    if result:
        
        print(f"\nFinal result: {result}")
        
