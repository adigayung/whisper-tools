import os
import time
import gc
from pydub import AudioSegment
import torchaudio
import torchaudio.transforms as T
import numpy as np
from libs.shared_model import whisper_model as model
from panns_inference import AudioTagging, labels
from libs.print_log import log_print

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def check_panns_model(file_path, panns_model):
    try:
        waveform, sr = torchaudio.load(file_path, normalize=True)
        if sr != 16000:
            waveform = T.Resample(sr, 16000)(waveform)
        audio = waveform[0].unsqueeze(0).numpy()
        clipwise_output, _ = panns_model.inference(audio)
        clipwise_output = np.squeeze(clipwise_output)
        topk = np.argsort(clipwise_output)[-5:][::-1]
        tags = [(labels[int(i)], float(clipwise_output[int(i)])) for i in topk]
        for tag, score in tags:
            if ("music" in tag.lower() or "sing" in tag.lower()) and score > 0.4:
                return True
        return False
    except Exception as e:
        log_print(f"[❌] Gagal deteksi dengan PANNs: {file_path} - {e}")
        return False

def is_music(audio_path, panns_model, language="id"):
    try:
        result = model.transcribe(audio_path, fp16=False)
        full_text = result.get("text", "").strip().lower()
        detected_lang = result.get("language", "und")
        log_print(f"Bahasa: {detected_lang} | Isi: {full_text}")

        if detected_lang != language:
            return True
        if full_text in ["thanks for watching!", "music"]:
            return True
        if full_text.count("music") >= 5:
            return True
        if full_text.startswith("sub indo by"):
            return True
        if check_panns_model(audio_path, panns_model):
            return True
        return False
    except Exception as e:
        log_print(f"[❌] Gagal transkripsi: {audio_path} - {e}")
        return True

def create_audio_preview(file_path, duration_seconds=8):
    try:
        audio = AudioSegment.from_file(file_path)
        preview = audio[:duration_seconds * 1000]

        # Simpan di folder yang sama dengan file asli
        dir_name = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        preview_path = os.path.join(dir_name, f"{base_name}_review.wav")

        # Pastikan foldernya ada
        os.makedirs(os.path.dirname(preview_path), exist_ok=True)

        preview.export(preview_path, format="wav")
        return preview_path
    except Exception as e:
        log_print(f"[❌] Gagal buat preview: {file_path} - {e}")
        return None


def detect_music_folder(path, language="id"):
    import torch
    audio_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(path)
        for f in files
        if os.path.splitext(f)[1].lower() in [".mp3", ".wav"]
    ]
    hasil = []
    panns_model = AudioTagging(device="cuda")
    for i, file in enumerate(audio_files, 1):
        log_print(f"[{i}/{len(audio_files)}] Proses: {os.path.basename(file)}")
        preview = create_audio_preview(file)
        if not preview:
            continue
        if is_music(preview, panns_model, language=language):
            os.remove(preview)
            os.remove(file)
            hasil.append((file, "Dihapus (Musik)"))
        else:
            os.remove(preview)
            hasil.append((file, "Disimpan (Bukan Musik)"))
    gc.collect()
    del panns_model
    torch.cuda.empty_cache()
    gc.collect()

    return hasil
