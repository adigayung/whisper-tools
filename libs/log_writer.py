import json
import os
from datetime import datetime

LOG_FILE = "processing_log.json"

def init_log():
    """Inisialisasi log kosong"""
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

def append_log(log_data):
    """Tambahkan satu log baru ke log file"""
    log_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **log_data
    }

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.append(log_entry)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
