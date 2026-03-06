from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path


def load_transcription(transcription_path: str) -> str | None:
    """Load plain transcription text from file."""
    path = Path(transcription_path)
    if not path.exists():
        print(f"Error: Transcription file not found: {path}")
        return None
    return path.read_text(encoding="utf-8").strip()


def load_prompt(prompt_path: str) -> str | None:
    """
    Load and extract only the AUDIO + CONTEXT fields from
    video_generation_prompt.txt for fair comparison.
    """
    path = Path(prompt_path)
    if not path.exists():
        print(f"Error: Prompt file not found: {path}")
        return None

    lines = path.read_text(encoding="utf-8").splitlines()
    extracted = []

    for line in lines:
        line = line.strip()
        # Only extract AUDIO and CONTEXT lines for semantic comparison
        if line.startswith("AUDIO:") or line.startswith("CONTEXT:"):
            content = line.split(":", 1)[1].strip()
            if content:
                extracted.append(content)

    return " ".join(extracted)


def calculate_semantic_score(
    transcription_path: str = "transcription_output/transcription_plain.txt",
    prompt_path: str        = "transcription_output/video_generation_prompt.txt",
    output_dir: str         = "transcription_output"
) -> float | None:
    """
    Calculate semantic similarity between the original video transcription
    and the generated video prompt using TF-IDF cosine similarity.

    Args:
        transcription_path (str): Path to transcription_plain.txt
        prompt_path (str):        Path to video_generation_prompt.txt
        output_dir (str):         Directory to save the score report

    Returns:
        float: Similarity score between 0 and 1
    """
    try:
        # Load both texts
        print("Loading transcription...")
        transcription = load_transcription(transcription_path)
        if not transcription:
            return None

        print("Loading video generation prompt...")
        prompt_text = load_prompt(prompt_path)
        if not prompt_text:
            return None

        # Calculate TF-IDF cosine similarity
        print("Calculating semantic similarity...")
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([transcription, prompt_text])
        score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        score = round(float(score), 4)

        print(f"Semantic Similarity Score: {score}")
        print(f"Accuracy: {round(score * 100, 2)}%")

        # Save report
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        report_path = output / "semantic_score_report.txt"
        with report_path.open("w", encoding="utf-8") as f:
            f.write("SEMANTIC SIMILARITY REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Transcription file: {transcription_path}\n")
            f.write(f"Prompt file:        {prompt_path}\n\n")
            f.write(f"Method:             TF-IDF Cosine Similarity\n\n")
            f.write("-" * 40 + "\n")
            f.write(f"Semantic Score:     {score}\n")
            f.write(f"Accuracy:           {round(score * 100, 2)}%\n\n")
            f.write("-" * 40 + "\n\n")
            f.write("WHAT THIS MEANS\n\n")
            f.write(interpret_score(score))

        print(f"Report saved to: {report_path}")
        return score

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def interpret_score(score: float) -> str:
    """Return a human-readable interpretation of the similarity score."""
    if score >= 0.8:
        return (
            "Excellent (>=80%): The prompt very accurately captures\n"
            "the content of the original video. The generated video\n"
            "should closely match the original.\n"
        )
    elif score >= 0.6:
        return (
            "Good (60-80%): The prompt captures most of the original\n"
            "video content. Minor details may differ in the generated video.\n"
        )
    elif score >= 0.4:
        return (
            "Moderate (40-60%): The prompt captures the general theme\n"
            "but misses some specific details from the original video.\n"
        )
    else:
        return (
            "Low (<40%): The prompt may not accurately represent the\n"
            "original video. Consider improving the scene description\n"
            "or transcription quality.\n"
        )


if __name__ == "__main__":
    score = calculate_semantic_score()
    if score is not None:
        print(f"\nFinal Semantic Score: {score} ({round(score * 100, 2)}%)")