import json
import os
from libs.utils import load_model_config

config = load_model_config()

use_faster = config.get("use_faster_whisper", False)
use_batch_mode = config.get("use_batch_mode", False)
model_is_batched = False  # <-- bisa dicek di kode lain jika ingin

if use_faster:
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    whisper_core_model = WhisperModel(
        model_size_or_path=config["model_size"],
        device=config["device"],
        compute_type=config["compute_type"]
    )

    if use_batch_mode:
        whisper_model = BatchedInferencePipeline(model=whisper_core_model)
        model_is_batched = True
    else:
        whisper_model = whisper_core_model
else:
    import whisper
    whisper_model = whisper.load_model(config["model_size"], device=config["device"])
