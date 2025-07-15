# File name : split_wav.py
import os
import glob
import random
import time
from pydub import AudioSegment
import torch
import gc
import re
from libs.log_writer import init_log, append_log
from libs.print_log import log_print
from libs.utils import load_model_config

def start_split_wav(form_data, debug=False, whisper_model=None):
    from libs.shared_model import get_whisper_model

    input_path = form_data.get("input_path")
    whisper_model_name = form_data.get("model")
    language_target = form_data.get("language_target")
    min_dur = float(form_data.get("min_dur", 1.0))
    max_dur = float(form_data.get("max_dur", 8.0))
    vad_threshold = float(form_data.get("vad_threshold", 0.5))
    shuffle_metadata = form_data.get("shuffle_metadata") == "on"

    log_print("Inisialisasi log...", "STEP", debug)
    init_log()

    if not os.path.isdir(input_path):
        log_print("❌ Input path tidak valid atau tidak ditemukan.", "ERROR", debug)
        return {
            "status": "error",
            "message": "Folder tidak ditemukan!",
            "inc": 1
        }

    audio_files = []
    for ext in ("*.wav", "*.mp3"):
        audio_files.extend(glob.glob(os.path.join(input_path, ext)))

    audio_files = sorted(audio_files, key=lambda x: natural_keys(os.path.basename(x)))

    total = len(audio_files)
    log_print(f"Total file audio ditemukan: {total}", "INFO", debug)

    output_path = os.path.join(input_path, "segments")
    os.makedirs(output_path, exist_ok=True)

    log_print(f"Memuat model Whisper: {whisper_model_name}", "STEP", debug)
    #whisper_model = get_whisper_model(whisper_model_name)

    log_print("Memuat model VAD Silero...", "STEP", debug)
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
    get_speech_timestamps, _, read_audio, _, _ = utils

    counter = 0
    raw_csv_files = []
    cfg = load_model_config()

    skipped_log_path = os.path.join(input_path, "skipped_segments.txt")
    open(skipped_log_path, "w", encoding="utf-8").close()

    for idx, audio_file in enumerate(audio_files, 1):
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        log_print(f"[{idx}/{total}] Memproses: {base_name}", "STEP", debug)

        start_time = time.time()
        wav = read_audio(audio_file, sampling_rate=16000)
        speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000, threshold=vad_threshold)

        if not speech_timestamps:
            log_print(f"   ✘ Tidak ada segment terdeteksi, dilewatkan", "WARNING", debug)
            continue

        audio = AudioSegment.from_file(audio_file)
        metadata_path_per_raw = os.path.join(output_path, f"{base_name}.csv")
        raw_csv_files.append(metadata_path_per_raw)

        for ts in speech_timestamps:
            start_time_per_segment = time.time()
            start_ms = int(ts['start'] / 16)
            end_ms = int(ts['end'] / 16)
            duration_sec = (end_ms - start_ms) / 1000
            filename = f"{base_name}_{counter:04d}.wav"
            if min_dur <= duration_sec <= max_dur:
                segment = audio[start_ms:end_ms]                
                path = os.path.join(output_path, filename)
                segment.export(path, format="wav")

                try:
                    if cfg["use_faster_whisper"]:
                        if cfg.get("use_batch_mode", False):
                            segments, info = whisper_model.transcribe(path, language=language_target, batch_size=cfg["batch_size"])
                        else:
                            segments, info = whisper_model.transcribe(path, language=language_target)

                        segments = list(segments)
                        full_text = " ".join([s.text.strip() for s in segments])
                        result = {
                            "text": full_text,
                            "segments": segments,
                            "language": info.language,
                            "duration": info.duration
                        }
                    else:
                        result = whisper_model.transcribe(path, language=language_target)
                except Exception as e:
                    os.remove(path)
                    log_print(f"      ✘ Gagal transkripsi: {e}", "ERROR", debug)
                    continue

                if isinstance(result, tuple):
                    segments, info = result
                    text = " ".join([s.text.strip() for s in segments]).strip()
                    detected_lang = info.language
                else:
                    text = result["text"].strip()
                    detected_lang = result.get("language", "unknown")

                if not text:
                    os.remove(path)
                    log_print(f"      ✘ Transkripsi kosong, file dihapus", "WARNING", debug)
                    msg = f"{filename}|durasi {duration_sec:.2f}s Transkripsi kosong"
                    with open(skipped_log_path, "a", encoding="utf-8") as f:
                        f.write(msg + "\n")
                    continue

                if len(text) <= 3:
                    os.remove(path)
                    log_print(f"      ✘ Transkripsi terlalu pendek ('{text}'), file dihapus", "WARNING", debug)
                    msg = f"{filename}|{text}|durasi {duration_sec:.2f}s Transkripsi terlalu pendek"
                    with open(skipped_log_path, "a", encoding="utf-8") as f:
                        f.write(msg + "\n")
                    continue

                line = f"{filename}|{text}"
                with open(metadata_path_per_raw, "a", encoding="utf-8") as f_meta:
                    f_meta.write(line + "\n")

                preview = text[:60].strip().replace('\n', ' ')
                log_print(f"      ✔ Transkripsi: {preview}...", "SUCCESS", debug)
                start_time_per_segment_end = time.time()
                log_print(f"   ↳ Segment {filename} | Durasi: {duration_sec:.2f}s | Processing time: {start_time_per_segment_end - start_time_per_segment:.2f}s ", "INFO", debug)
                counter += 1
            else:
                log_print(f"   ✘ Lewatkan segment (durasi {duration_sec:.2f}s di luar rentang)", "WARNING", debug)
                msg = f"{filename}|(skip)|durasi {duration_sec:.2f}s tidak sesuai batas {min_dur}-{max_dur}s"
                with open(skipped_log_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")

        # SELESAI satu file
        elapsed = time.time() - start_time
        remaining = elapsed * (total - idx)
        log_print(f"   ✅ Selesai {base_name}: {len(speech_timestamps)} segmen, waktu: {round(elapsed, 2)} detik", "SUCCESS2", debug)

        log = {
            "file": base_name,
            "segments": len(speech_timestamps),
            "elapsed": round(elapsed, 2),
            "remaining": round(remaining, 2),
            "progress": f"{idx}/{total}"
        }
        append_log(log)
        log_print(f"Selesai {base_name}: {len(speech_timestamps)} segmen, waktu: {round(elapsed, 2)} detik", "SUCCESS", debug)

    # GABUNGKAN metadata
    merged_lines = []
    for csv_file in raw_csv_files:
        if os.path.isfile(csv_file):
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

    # CLEANUP

    # try:
    #     del whisper_model
    #     del vad_model
    #     del read_audio
    #     del get_speech_timestamps
    # except:
    #     pass

    # torch.cuda.empty_cache()
    # gc.collect()

    return {
        "status": "done",
        "files_processed": total,
        "segments_generated": counter,
        "inc": 1
    }

def natural_keys(text):
    # Fungsi split angka vs huruf untuk sorting alami
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]