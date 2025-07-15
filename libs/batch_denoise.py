import os
from libs.denoise import denoise
from libs.print_log import log_print
from df.enhance import init_df
import torch
import gc

def run_batch_denoise(input_folder, debug=False, is_replace=True):
    if not os.path.isdir(input_folder):
        log_print(f"❌ Folder tidak ditemukan: {input_folder}", "ERROR", debug)
        return {"error": f"Folder tidak ditemukan: {input_folder}"}

    files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".mp3", ".wav"))
    ])

    if not files:
        log_print("❌ Tidak ada file .wav ditemukan di folder.", "WARNING", debug)
        return {"error": "Tidak ada file .wav ditemukan di folder."}

    log_print(f"🎧 Menemukan {len(files)} file audio. Memulai proses denoise...", "INFO", debug)

    processed = 0
    df_model, df_state, _ = init_df(model_base_dir="DeepFilterNet3")
    for idx, file in enumerate(files, start=1):
        input_path = os.path.join(input_folder, file)
        file_base, _ = os.path.splitext(file)

        try:
            denoise(input_path, df_model, df_state, is_replace=is_replace)
            log_print(f"[{idx}/{len(files)}] ✔ Denoised: {file} -> {file_base}_denoised.wav", "SUCCESS", debug)
            processed += 1
        except Exception as e:
            log_print(f"[{idx}/{len(files)}] ❌ Gagal memproses {file}: {e}", "ERROR", debug)

    del df_model
    del df_state
    torch.cuda.empty_cache()
    gc.collect()

    log_print(f"🎯 Proses selesai. Total berhasil: {processed}/{len(files)}", "INFO", debug)

    return {
        "processed_files": processed,
        "total_files": len(files),
        "output_folder": input_folder
    }
