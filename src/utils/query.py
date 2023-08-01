import requests


API_ENDPOINT = "http://localhost:5000/api"


def make_airoboros(text):
    return f"""
Context: A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input.
Continue the chat dialogue below. Write a single reply for the character "ASSISTANT".

USER: {text}
ASSISTANT:
"""


DIVINE = {
    'temparature': 1.31,
    'repetition_penalty': 1.17,
    'top_k': 49.0,
    'top_p': 0.14,
    'typical_p': 1.0,
    'top_a': 0.52,
    'epsion_cutoff': 1.49,
    'eta_cutoff': 10.42,
}


def query_api(prompt, preset=None):
    url = f"{API_ENDPOINT}/v1/generate"

    if not preset:
        preset = DIVINE

    data = {**preset, "prompt": prompt, "truncation_length": 1024 * 16, "max_new_tokens": 1000}
    resp = requests.post(url, json=data)
    resp.raise_for_status()
    data = resp.json()["results"][0]["text"].strip()
    return data
