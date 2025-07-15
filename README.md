# 🎙️ whisper-tools

A collection of audio processing tools built around Whisper, Demucs, and standard audio manipulation libraries. Designed to streamline dataset preparation, audio normalization, music separation, and file management.

---

## ✨ Features

### 🎙️ 1. Splitter
Creates a dataset by slicing `.mp3` or `.wav` audio files into smaller chunks using Whisper transcription. Perfect for preparing data for TTS or ASR model training.

### 🎧 2. Normalize + Merge
Normalizes the loudness of audio files to a target level and merges them into a single file. This is essentially the reverse of the Splitter — useful for reconstructing longer audio content or preparing audio for playback.

### ⏱️ 3. Audio Duration Calculator
Calculates the total duration (in seconds and hours) and counts the number of audio files in a specified folder. Helpful for dataset validation and training time estimation.

### 🎼 4. Demucs Splitter
Uses the Demucs model to separate *vocals* and *music* into distinct files. Outputs include `vocal.wav` and `music.wav`.

### 🧹 5. Clean Music Files
Automatically deletes audio files that contain only music with no vocals.  
⚠️ **Warning: This tool is destructive. Use it with extreme caution, and always back up your data before running this process.**

---

## 📁 Folder Structure
Ensure that input/output folder structures are organized. Most tools expect `.mp3` or `.wav` files and save their results in `output` or `processed` subfolders.

---

## ⚙️ Requirements

- Python ≥ 3.10  
- `ffmpeg` installed and in PATH  
- Install dependencies:

```bash
git clone https://github.com/adigayung/whisper-tools.git
cd whisper-tools
pip install -r requirements.txt
```

🚀 Usage
All tools are available through a unified Flask web interface.

Start the application with:
```bash
python main.py
```
Then open your browser and go to:
```bash
http://localhost:5000
```
From the web UI, you can run each tool easily without touching the command line.

⚠️ Important Warning
The Clean Music Files tool will permanently delete files detected as containing music only. This feature is powerful but dangerous if misused. Always verify results manually and back up your data.

📌 Future Plans
Whisper + timestamp alignment

Multi-format audio support (FLAC, M4A, etc.)

Built-in audio preview

GUI improvements with drag & drop

Pull requests and suggestions are welcome! 💡

🌍 Run via Ngrok (Public Access)
To expose the Flask server publicly via Ngrok, use the --NGROK_AUTH_TOKEN= argument:
```bash
python main.py --NGROK_AUTH_TOKEN=your_ngrok_token_here
```
This will start the server and print a public Ngrok URL, such as:
```bash
https://abcd-1234.ngrok.io
```
You can get your Ngrok token from:
👉 [https://dashboard.ngrok.com/get-started/setup](https://dashboard.ngrok.com/get-started/setup)
