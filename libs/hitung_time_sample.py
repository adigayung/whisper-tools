import os
import sys
import wave
from tqdm import tqdm
from pydub import AudioSegment

def get_total_duration(folder_path):
    total_duration = 0.0
    audio_files = [f for f in os.listdir(folder_path) if f.endswith((".wav", ".mp3"))]

    for filename in tqdm(audio_files, desc="⏳ Menghitung durasi"):
        filepath = os.path.join(folder_path, filename)

        if filename.endswith(".wav"):
            with wave.open(filepath, 'r') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
        else:
            audio = AudioSegment.from_file(filepath)
            duration = len(audio) / 1000.0

        total_duration += duration

    return total_duration, len(audio_files)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hitung_time_sample.py <folder_path>")
        sys.exit(1)

    folder = sys.argv[1]
    total_seconds, total_files = get_total_duration(folder)
    print(f"\n📁 Total file audio: {total_files}")
    print(f"🕒 Total durasi: {total_seconds / 3600:.2f} jam")
