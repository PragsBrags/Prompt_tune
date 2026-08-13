import requests

OLLAMA_URL = "http://localhost:11434"

def translate(model_name, messages):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 128},
        },
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()
