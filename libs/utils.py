import json
import os
import subprocess

def load_model_config():
    config_path = os.path.join(os.getcwd(), "model.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        # Validasi kunci penting
        required_keys = {"device", "compute_type", "model_size", "use_faster_whisper"}
        if not required_keys.issubset(config.keys()):
            raise ValueError(f"model.json tidak lengkap. Diperlukan: {required_keys}")
        return config
    except Exception as e:
        raise RuntimeError(f"Gagal membaca 'model.json': {e}")

def convert_to_mono_22050(wav_path):
    temp_path = wav_path.replace(".wav", "_converted.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", wav_path,
        "-ac", "1",              # mono
        "-ar", "22050",          # 22050 Hz
        temp_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace(temp_path, wav_path)
    