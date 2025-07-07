import os
import subprocess
import shutil
from libs.print_log import log_print
from libs.utils import convert_to_mono_22050

def run_demucs_batch(input_folder, log_path="demucs_output.txt", selected_stems=None, debug=False, device_id="cuda"):
    import os
    import shutil
    import subprocess
    from libs.print_log import log_print

    if not os.path.isdir(input_folder):
        log_print(f"❌ Folder tidak ditemukan: {input_folder}", "ERROR", debug)
        return {"error": f"Folder tidak ditemukan: {input_folder}"}

    if not selected_stems:
        selected_stems = ["vocals"]

    files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".mp3", ".wav"))
    ])

    if not files:
        log_print("❌ Tidak ada file audio ditemukan di folder.", "WARNING", debug)
        return {"error": "Tidak ada file audio ditemukan di folder."}

    emoji = "🖥️" if device_id == "cpu" else "⚡"
    device_label = "CPU" if device_id == "cpu" else "GPU (CUDA)"
    log_print(f"{emoji} Menggunakan perangkat: {device_label} {device_id}", "INFO", debug)

    success_count = 0
    output_folder = os.path.join(input_folder, "output")
    os.makedirs(output_folder, exist_ok=True)

    log_print(f"🔎 Menemukan {len(files)} file audio, memulai proses demucs...", "INFO", debug)

    with open(log_path, "w", encoding="utf-8") as log_file:
        for idx, file in enumerate(files, start=1):
            full_path = os.path.join(input_folder, file)
            file_name = os.path.splitext(file)[0]
            log_print(f"[{idx}/{len(files)}] 🔄 Memproses: {file}", "INFO", debug)
            log_file.write(f"\n🔄 Memproses: {file}\n")

            try:
                cmd = [
                    "demucs",
                    full_path,
                    "--out", output_folder,
                    "--device", device_id,
                    "--shifts", "0",
                    "--overlap", "0.25"
                ]

                if selected_stems == ["vocals"]:
                    cmd += ["--two-stems", "vocals"]

                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    shell=False
                )
                log_file.write(result.stdout)

                separated_dir = os.path.join(output_folder, "htdemucs", file_name)
                if os.path.isdir(separated_dir):
                    for stem_file in os.listdir(separated_dir):
                        stem_name = os.path.splitext(stem_file)[0].lower()
                        if stem_name not in selected_stems:
                            continue
                        stem_src = os.path.join(separated_dir, stem_file)
                        new_name = f"{file_name}_{stem_name}.wav"
                        stem_dst = os.path.join(output_folder, new_name)
                        shutil.move(stem_src, stem_dst)
                        convert_to_mono_22050(stem_dst)
                        log_print(f"      ✔ Hasil: {new_name}", "SUCCESS", debug)
                    shutil.rmtree(separated_dir, ignore_errors=True)
                    success_count += 1
                else:
                    log_print("❌ Tidak ditemukan folder hasil demucs!", "ERROR", debug)

            except Exception as e:
                log_print(f"      ❌ Exception saat memproses {file}: {e}", "ERROR", debug)
                log_file.write(f"❌ Exception: {e}\n")

    htdemucs_folder = os.path.join(output_folder, "htdemucs")
    if os.path.exists(htdemucs_folder):
        shutil.rmtree(htdemucs_folder, ignore_errors=True)

    log_print(f"🎯 Proses selesai. Total berhasil: {success_count}/{len(files)}", "INFO", debug)

    if success_count == 0:
        return {
            "error": "Tidak ada file berhasil diproses.",
            "log_file": log_path
        }

    return {
        "processed_files": success_count,
        "total_files": len(files),
        "output_folder": output_folder,
        "log_file": log_path
    }
