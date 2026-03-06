import time
import json
from pathlib import Path
from google import genai
from google.genai import types


def generate_video(
    prompt_file: str = "transcription_output/video_generation_prompt.txt",
    output_dir: str = "transcription_output",
    api_key: str = None
) -> str | None:
    """
    Read the merged video generation prompt and generate a video using Google Veo.

    Args:
        prompt_file (str): Path to the video_generation_prompt.txt file
        output_dir (str): Directory to save the generated video
        api_key (str): Google Gemini API key

    Returns:
        str: Path to the saved video, or None if failed
    """
    try:
        # Load the prompt
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            print(f"Error: Prompt file not found: {prompt_path}")
            return None

        prompt_text = prompt_path.read_text(encoding="utf-8")
        print("Prompt loaded successfully.")

        # Set up Gemini client
        client = genai.Client(api_key=api_key)

        print("Sending prompt to Veo for video generation...")
        print("This may take several minutes...")

        # Start video generation job
        operation = client.models.generate_videos(
            model="veo-2.0-generate-001",
            prompt=prompt_text,
            config=types.GenerateVideosConfig(
                aspect_ratio="16:9",
                number_of_videos=1,
            )
        )

        # Poll until done
        while not operation.done:
            print("Waiting for video generation to complete...")
            time.sleep(10)
            operation = client.operations.get(operation)

        # Save the generated video
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        for i, video in enumerate(operation.response.generated_videos):
            video_path = output / f"generated_video_{i + 1}.mp4"
            client.files.download(file=video.video)
            video.video.save(str(video_path))
            print(f"Video saved to: {video_path}")
            return str(video_path)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    api_key = "AIzaSyDuJ8QPWCV5TlHJwwrw1yc684qJUbtVAUY"
    result = generate_video(api_key=api_key)
    if result:
        print(f"\nDone! Generated video: {result}")
