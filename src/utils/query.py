import requests


API_ENDPOINT = "http://localhost:5000/api"
"""
Context: A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input.
# Continue the chat dialogue below. Write a single reply for the character "ASSISTANT".
Continue the chat dialogue below, with the "ASSISTANT" providing a single, brief reply.
"""


def make_airoboros(text, end=' '):
    return f"""
Context: A chat between a user and an assistant. The assistant provides concise summaries of the user's text, distilling it down to the most important details in approximately 30 words.

USER: {text}
ASSISTANT:{end}
"""


DIVINE = {
    'temperature': 0.7,
    'repetition_penalty': 1.3,
    'repetition_penalty_range': 128,
    'top_k': 33.0,
    'top_p': 0.6,
    'typical_p': 1.0,
    'top_a': 0.0,
    #'epsion_cutoff': 1.49,
    #'eta_cutoff': 10.42,
    'length_penalty': 0.8,
}


def query_api(prompt, preset=None, overrides=None) -> str:
    url = f"{API_ENDPOINT}/v1/generate"

    if not preset:
        preset = DIVINE
    preset = preset.copy()

    if not overrides:
        overrides = {}
    for k, v in overrides.items():
        if k in preset:
            preset[k] += v
        else:
            preset[k] = v
    preset['temperature'] = max(preset['temperature'], 0.01)
    data = {**preset, "prompt": prompt, "max_new_tokens": 1000}
    resp = requests.post(url, json=data)
    resp.raise_for_status()
    data = resp.json()["results"][0]["text"].strip()
    return data
