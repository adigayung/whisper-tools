import os
import shutil
import subprocess
from pydub import AudioSegment
from pydub.utils import which
from libs.print_log import log_print

AudioSegment.converter = which("ffmpeg")
target_lufs = "-20"

def normalize_and_merge(input_folder, normalize=True, debug=False):
    if not os.path.isdir(input_folder):
        log_print(f"❌ Folder tidak ditemukan: {input_folder}", "ERROR", debug)
        return {"error": f"Folder tidak ditemukan: {input_folder}"}

    normalized_folder = os.path.join(input_folder, "hasil_enhanced_loud")
    os.makedirs(normalized_folder, exist_ok=True)

    if normalize:
        log_print("🎚️ Normalisasi audio dimulai...", "STEP", debug)
        for filename in sorted(os.listdir(input_folder)):
            if filename.lower().endswith((".mp3", ".wav")):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(normalized_folder, os.path.splitext(filename)[0] + ".wav")
                cmd = [
                    "ffmpeg", "-y",
                    "-i", input_path,
                    "-af", f"loudnorm=I={target_lufs}:LRA=7:TP=-2",
                    "-ar", "22050", "-ac", "1", "-sample_fmt", "s16",
                    output_path
                ]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    log_print(f"✔ Berhasil normalisasi: {filename}", "SUCCESS", debug)
                except subprocess.CalledProcessError as e:
                    log_print(f"⚠️ Gagal normalisasi: {filename} | {e}", "WARNING", debug)
    else:
        log_print("📎 Lewatkan normalisasi, salin file apa adanya...", "STEP", debug)
        for filename in sorted(os.listdir(input_folder)):
            if filename.lower().endswith((".mp3", ".wav")):
                src = os.path.join(input_folder, filename)
                dst = os.path.join(normalized_folder, filename)
                try:
                    shutil.copy2(src, dst)
                    log_print(f"✔ Disalin: {filename}", "INFO", debug)
                except Exception as e:
                    log_print(f"⚠️ Gagal menyalin {filename}: {e}", "WARNING", debug)

    # Gabungkan hasil
    combined = AudioSegment.empty()
    success_files = []

    for file in sorted(os.listdir(normalized_folder)):
        if file.lower().endswith((".mp3", ".wav")):
            path = os.path.join(normalized_folder, file)
            try:
                log_print(f"🔍 Membaca file: {file}", "STEP", debug)
                audio = AudioSegment.from_file(path)
                log_print(f"📊 {file}: {audio.frame_rate} Hz | {audio.channels} ch | {audio.sample_width*8}-bit", "STEP", debug)
                duration = len(audio) / 1000.0  # detik
                log_print(f"⏱️ Durasi {file}: {duration:.2f} detik", "INFO", debug)

                if duration > 0.1:
                    combined += audio
                    success_files.append(path)
                    log_print(f"➕ Ditambahkan ke gabungan: {file}", "STEP", debug)
                else:
                    log_print(f"⚠️ Durasi file terlalu pendek: {file}", "WARNING", debug)

            except Exception as e:
                log_print(f"⚠️ Gagal membaca {file}: {e}", "WARNING", debug)

    if success_files:
        output_path = os.path.join(normalized_folder, "file_baru.wav")
        combined.export(output_path, format="wav")
        log_print(f"✅ File gabungan berhasil dibuat: {output_path}", "SUCCESS", debug)

        for path in success_files:
            if os.path.basename(path) != "file_baru.wav":
                try:
                    os.remove(path)
                except Exception as e:
                    log_print(f"⚠️ Gagal menghapus {path}: {e}", "WARNING", debug)

        return {
            "normalized_folder": normalized_folder,
            "output_path": output_path,
            "file_count": len(success_files)
        }
    else:
        log_print("❌ Tidak ada file berhasil digabung.", "ERROR", debug)
        return {"error": "Tidak ada file berhasil diproses."}
