# file: main.py
from flask import Flask, request, render_template, jsonify, redirect, session
import json
import os
import sys
import time
import threading
from libs.utils import find_free_port
from libs.split_wav import start_split_wav
from libs.hitung_time_sample import get_total_duration
from libs.normalize_merge import normalize_and_merge
from libs.demucs_misc import run_demucs_batch
from libs.detect_music import detect_music_folder
from libs.batch_denoise import run_batch_denoise
from libs.check_metadata import check_metadata_vs_files
import torch
import gc

WHISPER_MODELS = None
WHISPER_MODELS_LOAD_COUNT = 0
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
    from libs.shared_model import get_whisper_model
    global WHISPER_MODELS, WHISPER_MODELS_LOAD_COUNT
    if WHISPER_MODELS_LOAD_COUNT == 0:
        WHISPER_MODELS_LOAD_COUNT = WHISPER_MODELS_LOAD_COUNT + 1
        whisper_model_name = request.form.get("model")
        WHISPER_MODELS = get_whisper_model(whisper_model_name)
    
    result = start_split_wav(request.form, DEBUG_MODE, whisper_model=WHISPER_MODELS)
    WHISPER_MODELS_LOAD_COUNT = max(0, WHISPER_MODELS_LOAD_COUNT - result.get("inc", 0))

    if WHISPER_MODELS_LOAD_COUNT == 0:
        try:
            del WHISPER_MODELS
            WHISPER_MODELS = None
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
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
        use_denoise = request.form.get("use_denoise") == "1"

        # Simpan ke session atau kirim via redirect
        session["demucs_data"] = {
            "folder_path": folder_path,
            "selected_stems": selected_stems,
            "device_id": device_id,
            "use_denoise": use_denoise
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
        use_denoise=data["use_denoise"],
        debug=DEBUG_MODE
    )

    return render_template("demucs_result.html", result=result)

@app.route("/Denoise", methods=["GET", "POST"])
def denoise_page():
    if request.method == "POST":
        folder = request.form.get("folder_path")
        if not os.path.isdir(folder):
            return render_template("denoise.html", error="Folder tidak ditemukan!")

        result = run_batch_denoise(folder, debug=True, is_replace=False)  # ✅ Panggil fungsinya langsung

        if "error" in result:
            return render_template("denoise.html", error=result["error"])

        return render_template("denoise.html", result=f"Berhasil memproses {result['processed_files']} dari {result['total_files']} file.")

    return render_template("denoise.html")

@app.route("/check_metadata", methods=["GET", "POST"])
def check_metadata():
    result = None
    if request.method == "POST":
        metadata_path = request.form.get("metadata_path", "").strip()
        wav_path = request.form.get("wav_path", "").strip()

        if os.path.isfile(metadata_path) and os.path.isdir(wav_path):
            result = check_metadata_vs_files(metadata_path, wav_path)
        else:
            result = {"error": "Path metadata atau folder WAV tidak valid."}

    return render_template("metadata_checker.html", result=result)
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--NGROK_AUTH_TOKEN", type=str, default=None, help="Ngrok Auth Token (optional)")
    args = parser.parse_args()

    port = find_free_port()

    if args.NGROK_AUTH_TOKEN:
        try:
            from pyngrok import ngrok, conf
        except ImportError:
            print("❌ Modul pyngrok belum terinstall. Jalankan: pip install pyngrok")
            exit(1)

        conf.get_default().auth_token = args.NGROK_AUTH_TOKEN
        public_url = ngrok.connect(port)
        print(f"🔗 Flask berjalan via ngrok: {public_url}")
    else:
        print(f"🌐 Flask berjalan di lokal: http://127.0.0.1:{port}")

    app.run(port=port, debug=FLASK_DEBUG_MODE, threaded=True)