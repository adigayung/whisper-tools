from flask import Flask, request, render_template, jsonify, redirect, session
import json
import os
import sys
import time
import threading
from libs.split_wav import start_split_wav
from libs.hitung_time_sample import get_total_duration
from libs.normalize_merge import normalize_and_merge
from libs.demucs_misc import run_demucs_batch
from libs.detect_music import detect_music_folder

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "whispertools2025")


MODEL_OPTIONS = [
    "tiny", "base", "small", "medium", "large",
    "large-v1", "large-v2", "large-v3", "turbo"
]
DEBUG_MODE = True
FLASK_DEBUG_MODE = False

@app.route("/", methods=["GET"])
def index():
    return render_template("splitter.html", models=MODEL_OPTIONS)

@app.route("/log", methods=["GET"])
def get_log():
    try:
        with open("processing_log.json", "r") as f:
            data = json.load(f)
            return jsonify(data)
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify([])  # return empty list instead of error

@app.route("/process", methods=["POST"])
def process():
    result = start_split_wav(request.form, DEBUG_MODE)
    return jsonify(result)

@app.route("/audiodurationcalculator", methods=["GET", "POST"])
def audiodurationcalculator():
    if request.method == "POST":
        folder_path = request.form.get("folder_path")

        if not os.path.exists(folder_path):
            return render_template("duration_form.html", error="Folder tidak ditemukan!")

        try:
            duration, file_count = get_total_duration(folder_path)
            return render_template("duration_form.html", folder=folder_path, duration=duration, file_count=file_count)
        except Exception as e:
            return render_template("duration_form.html", error=f"Terjadi kesalahan: {str(e)}")

    return render_template("duration_form.html")

@app.route("/normalize-merge", methods=["GET", "POST"])
def normalize_merge():
    result = None
    if request.method == "POST":
        folder_path = request.form.get("folder_path")
        normalize_flag = request.form.get("normalize") == "on"  # Checkbox bernilai "on" jika dicentang
        result = normalize_and_merge(folder_path, normalize=normalize_flag, debug=DEBUG_MODE)
    return render_template("normalize_merge.html", result=result)


@app.route('/restart', methods=['GET'])
def restart():
    def delayed_restart():
        time.sleep(1)  # kasih waktu agar template dikirim dulu
        python = sys.executable
        os.execv(python, [python] + sys.argv)

    threading.Thread(target=delayed_restart).start()
    return render_template("restart.html")

@app.route("/demucs", methods=["GET", "POST"])
def demucs():
    if request.method == "POST":
        folder_path = request.form.get("folder_path")
        selected_stems = request.form.getlist("stems")
        device_id = request.form.get("device", "cuda")

        # Simpan ke session atau kirim via redirect
        session["demucs_data"] = {
            "folder_path": folder_path,
            "selected_stems": selected_stems,
            "device_id": device_id
        }

        return redirect("/demucs-progress")

    return render_template("demucs.html")

@app.route("/demucs-progress")
def demucs_progress():
    data = session.pop("demucs_data", None)
    if not data:
        return redirect("/demucs")

    result = run_demucs_batch(
        input_folder=data["folder_path"],
        selected_stems=data["selected_stems"],
        device_id=data["device_id"],
        debug=DEBUG_MODE
    )

    return render_template("demucs_result.html", result=result)

@app.route("/detect_music", methods=["GET", "POST"])
def detect_music():
    results = []
    selected_lang = "id"
    if request.method == "POST":
        path = request.form.get("path")
        selected_lang = request.form.get("language") or "id"
        
        if os.path.isdir(path):
            results = detect_music_folder(path, language=selected_lang)

    return render_template("detect_music.html", results=results, selected_lang=selected_lang)


if __name__ == "__main__":
    app.run(debug=FLASK_DEBUG_MODE, threaded=True)
