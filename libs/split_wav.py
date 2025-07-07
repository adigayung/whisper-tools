import os
import glob
import random
import time
from pydub import AudioSegment
from libs.shared_model import whisper_model as model
import torch
from libs.log_writer import init_log, append_log
from libs.print_log import log_print
from libs.utils import load_model_config

def start_split_wav(form_data, debug=False):
    input_path = form_data.get("input_path")
    whisper_model_name = form_data.get("model")
    language_target = form_data.get("language_target")
    min_dur = float(form_data.get("min_dur", 1.0))
    max_dur = float(form_data.get("max_dur", 8.0))
    vad_threshold = float(form_data.get("vad_threshold", 0.5))
    shuffle_metadata = form_data.get("shuffle_metadata") == "on"

    log_print("Inisialisasi log...", "STEP", debug)
    init_log()

    audio_files = glob.glob(os.path.join(input_path, "*.wav")) + glob.glob(os.path.join(input_path, "*.mp3"))
    total = len(audio_files)
    log_print(f"Total file audio ditemukan: {total}", "INFO", debug)

    output_path = os.path.join(input_path, "segments")
    os.makedirs(output_path, exist_ok=True)

    log_print(f"Memuat model Whisper: {whisper_model_name}", "STEP", debug)
    whisper_model = model # whisper.load_model(whisper_model_name, device="cuda")

    log_print("Memuat model VAD Silero...", "STEP", debug)
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
    get_speech_timestamps, _, read_audio, _, _ = utils

    counter = 0
    raw_csv_files = []

    for idx, audio_file in enumerate(audio_files, 1):
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        log_print(f"[{idx}/{total}] Memproses: {base_name}", "STEP", debug)

        start_time = time.time()
        wav = read_audio(audio_file, sampling_rate=16000)
        speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000, threshold=vad_threshold)

        audio = AudioSegment.from_file(audio_file)
        metadata_path_per_raw = os.path.join(output_path, f"{base_name}.csv")
        raw_csv_files.append(metadata_path_per_raw)

        for ts in speech_timestamps:
            start_ms = int(ts['start'] / 16)
            end_ms = int(ts['end'] / 16)
            duration_sec = (end_ms - start_ms) / 1000

            if min_dur <= duration_sec <= max_dur:
                segment = audio[start_ms:end_ms]
                filename = f"{base_name}_{counter:04d}.wav"
                path = os.path.join(output_path, filename)
                segment.export(path, format="wav")

                log_print(f"   ↳ Segment {filename} | Durasi: {duration_sec:.2f}s", "INFO", debug)

                cfg = load_model_config()

                if cfg["use_faster_whisper"]:
                    if cfg.get("use_batch_mode", False):
                        segments, info = whisper_model.transcribe(path, language=language_target, batch_size=cfg["batch_size"])
                    else:
                        segments, info = whisper_model.transcribe(path, language=language_target)
                    
                    segments = list(segments)  # HARUS dijalankan
                    full_text = " ".join([s.text.strip() for s in segments])
                    result = {
                        "text": full_text,
                        "segments": segments,
                        "language": info.language,
                        "duration": info.duration
                    }
                else:
                    result = whisper_model.transcribe(path, language=language_target)

                if isinstance(result, tuple):
                    # Untuk faster_whisper
                    segments, info = result
                    text = " ".join([s.text.strip() for s in segments]).strip()
                    detected_lang = info.language
                else:
                    # Untuk whisper biasa
                    text = result["text"].strip()
                    detected_lang = result.get("language", "unknown")

                if not text:
                    os.remove(path)
                    log_print(f"      ✘ Transkripsi kosong, file dihapus", "WARNING", debug)
                    continue

                if len(text) <= 2:
                    os.remove(path)
                    log_print(f"      ✘ Transkripsi terlalu pendek ('{text}'), file dihapus", "WARNING", debug)
                    continue

                # Jika valid
                line = f"{filename}|{text}"
                with open(metadata_path_per_raw, "a", encoding="utf-8") as f_meta:
                    f_meta.write(line + "\n")

                preview = text[:60].strip().replace('\n', ' ')
                log_print(f"      ✔ Transkripsi: {preview}...", "SUCCESS", debug)

                counter += 1
            else:
                log_print(f"   ✘ Lewatkan segment (durasi {duration_sec:.2f}s di luar rentang)", "WARNING", debug)

            log_print(f"   ✅ Selesai {base_name}: {len(speech_timestamps)} segmen, waktu: {round(time.time() - start_time, 2)} detik", "SUCCESS2", debug)

        elapsed = time.time() - start_time
        remaining = elapsed * (total - idx)

        log = {
            "file": base_name,
            "segments": len(speech_timestamps),
            "elapsed": round(elapsed, 2),
            "remaining": round(remaining, 2),
            "progress": f"{idx}/{total}"
        }
        append_log(log)

        log_print(f"Selesai {base_name}: {len(speech_timestamps)} segmen, waktu: {round(elapsed, 2)} detik", "SUCCESS", debug)

    # Gabungkan metadata
    merged_lines = []
    for csv_file in raw_csv_files:
        if not os.path.isfile(csv_file):
            continue
        with open(csv_file, "r", encoding="utf-8") as f:
            merged_lines.extend(f.readlines())

    if not merged_lines:
        log_print("Tidak ada metadata yang berhasil digabungkan.", "WARNING", debug)

    if shuffle_metadata:
        log_print("Mengacak urutan metadata...", "INFO", debug)
        random.shuffle(merged_lines)

    metadata_path = os.path.join(output_path, "metadata.csv")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.writelines(merged_lines)

    log_print(f"Metadata akhir ditulis ke: {metadata_path}", "SUCCESS", debug)

    return {
        "status": "done",
        "files_processed": total,
        "segments_generated": counter
    }