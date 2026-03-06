from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

from services.speechtotext import transcribe_video
from services.scenedescribing import describe_video

FRONTEND_FOLDER = os.path.join(os.path.dirname(__file__), "../frontend")

app = Flask(__name__, static_folder=FRONTEND_FOLDER)
CORS(app)

UPLOAD_FOLDER = "uploads"
GROQ_API_KEY  = "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"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return send_from_directory(FRONTEND_FOLDER, "index.html")

@app.route("/result.html")
def result():
    return send_from_directory(FRONTEND_FOLDER, "result.html")


def compute_semantic_score_inline(plain_text: str, merged: list) -> float | None:
    """
    Compute TF-IDF cosine similarity directly from in-memory data.
    Compares the plain transcription text against all AUDIO + CONTEXT
    fields from the merged segments — same logic as semanticscore.py
    but without needing any files on disk.
    """
    try:
        if not plain_text or not merged:
            return None

        # Build the "prompt side" text from AUDIO + CONTEXT of each segment
        prompt_parts = []
        for seg in merged:
            if seg.get("text"):
                prompt_parts.append(seg["text"])
            if seg.get("context") and seg["context"] != "N/A":
                prompt_parts.append(seg["context"])

        prompt_text = " ".join(prompt_parts)

        if not prompt_text.strip():
            return None

        vectorizer   = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([plain_text, prompt_text])
        score        = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        return round(float(score), 4)

    except Exception as e:
        print("Inline semantic score error:", e)
        return None


@app.route("/upload", methods=["POST"])
def upload_video():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video uploaded"}), 400

        video = request.files["video"]
        video_path = os.path.join(UPLOAD_FOLDER, video.filename)
        video.save(video_path)
        print("Video saved:", video_path)

        # ── Step 1: Speech to text ──
        transcription = transcribe_video(video_path)
        segments      = transcription.get("segments", [])
        plain_text    = transcription.get("plain_text", "")   # already built by speechtotext.py

        formatted = []
        for seg in segments:
            formatted.append({
                "start": seg["start"],
                "end":   seg["end"],
                "text":  seg["text"]
            })
        print("Transcription done:", len(formatted), "segments")

        # ── Step 2: Scene describing ──
        print("Running scene analysis...")
        scenes = describe_video(video_path, api_key=GROQ_API_KEY, interval_seconds=4)
        print("Scene analysis done:", len(scenes) if scenes else 0, "frames")

        # ── Step 3: Merge ──
        merged = []
        for seg in formatted:
            seg_start = seg["start"]
            seg_end   = seg["end"]

            if scenes:
                matching = [s for s in scenes if seg_start <= s["timestamp"] < seg_end]
                if not matching:
                    matching = [min(scenes, key=lambda s: abs(s["timestamp"] - seg_start))]
                scene = matching[0]["semantic"]
            else:
                scene = {}

            merged.append({
                "start":      seg_start,
                "end":        seg_end,
                "text":       seg["text"],
                "action":     scene.get("action",     "N/A"),
                "camera":     scene.get("camera",     "N/A"),
                "emotion":    scene.get("emotion",    "N/A"),
                "atmosphere": scene.get("atmosphere", "N/A"),
                "context":    scene.get("context",    "N/A"),
            })

        # ── Step 4: Semantic Score — in memory, no files needed ──
        print("Calculating semantic score...")
        raw_score      = compute_semantic_score_inline(plain_text, merged)
        semantic_score = round(raw_score * 100, 1) if raw_score is not None else None
        print("Semantic score:", semantic_score)

        return jsonify({
            "transcript":     merged,
            "semantic_score": semantic_score
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Processing failed", "details": str(e)}), 500


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"


if __name__ == "__main__":
    app.run(debug=True, port=5000)
