import os

def parse_metadata_file(metadata_path):
    entries = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                filename = line.split("|")[0].strip()
                filename = os.path.splitext(filename)[0]  # hapus .wav
                entries.append(filename)
    return set(entries)

def list_wav_files(wav_path):
    return set(
        os.path.splitext(f)[0]
        for f in os.listdir(wav_path)
        if f.lower().endswith(".wav")
    )

def check_metadata_vs_files(metadata_path, wav_path):
    metadata_entries = parse_metadata_file(metadata_path)
    wav_files = list_wav_files(wav_path)

    missing_in_folder = metadata_entries - wav_files
    missing_in_metadata = wav_files - metadata_entries

    return {
        "total_metadata": len(metadata_entries),
        "total_wav": len(wav_files),
        "missing_in_folder": sorted(missing_in_folder),
        "missing_in_metadata": sorted(missing_in_metadata),
    }
