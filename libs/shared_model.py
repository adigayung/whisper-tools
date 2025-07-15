# file name : shared_model.py
import whisper
from faster_whisper import WhisperModel, BatchedInferencePipeline
from libs.utils import load_model_config

config = load_model_config()

def get_whisper_model(override_model_name=None):
    if override_model_name:
        print(f"Load Model Name : {override_model_name}")
    model_name = override_model_name or config["model_size"]
    device = config["device"]
    use_faster = config.get("use_faster_whisper", False)
    use_batch = config.get("use_batch_mode", False)
    
    if use_faster:
        core_model = WhisperModel(model_size_or_path=model_name, device=device, compute_type=config["compute_type"])
        return BatchedInferencePipeline(core_model) if use_batch else core_model
    else:
        return whisper.load_model(model_name, device=device)

# ✅ Untuk backward compatibility agar `from shared_model import whisper_model` tidak error
whisper_model = None
