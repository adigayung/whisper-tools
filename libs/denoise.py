import os
import torch
import subprocess
import soundfile as sf
import librosa
import uuid
from df.enhance import enhance, init_df


def convert_to_wav_ffmpeg(input_path, temp_wav_path):
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "24000", "-ac", "1",
        temp_wav_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not os.path.exists(temp_wav_path) or result.returncode != 0:
        print(f"❌ FFmpeg gagal konversi: {result.stderr.decode()}")
        return False
    return True


def denoise(input_path, model, df_state, is_replace=True):
    if not os.path.exists(input_path):
        print(f"❌ File tidak ditemukan: {input_path}")
        return False

    input_dir = os.path.dirname(input_path)
    input_name = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1].lower()

    output_dir = input_dir if is_replace else os.path.join(input_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{input_name}.wav")
    temp_wav = os.path.join(output_dir, f"temp_{uuid.uuid4().hex}.wav")

    if not convert_to_wav_ffmpeg(input_path, temp_wav):
        return False

    try:
        audio, _ = librosa.load(temp_wav, sr=24000)
        os.remove(temp_wav)
    except Exception as e:
        print(f"❌ Gagal load temp WAV: {e}")
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        return False

    audio_48k = librosa.resample(audio, orig_sr=24000, target_sr=48000)
    audio_tensor = torch.from_numpy(audio_48k).float().unsqueeze(0).to("cpu")

    enhanced = enhance(model, df_state, audio_tensor)
    enhanced_np = enhanced.squeeze().cpu().numpy()
    enhanced_24k = librosa.resample(enhanced_np, orig_sr=48000, target_sr=24000)

    save_path = input_path if is_replace and input_path.lower().endswith(".wav") else output_path
    sf.write(save_path, enhanced_24k, samplerate=24000)
    print(f"✅ Output disimpan di: {save_path}")
    return True
