import requests

OLLAMA_URL = "http://localhost:11434"

def load_model(model_cfg):
    response = requests.get(f"{OLLAMA_URL}/api/tags")
    response.raise_for_status()
    available = [m["name"] for m in response.json()["models"]]

    if model_cfg.name not in available:
        raise RuntimeError(
            f"Model '{model_cfg.name}' not found in Ollama. Run: ollama pull {model_cfg.name}"
        )

    return model_cfg.name
