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
    'temparature': 0.7,
    'repetition_penalty': 1.17,
    'top_k': 35.0,
    'top_p': 0.6,
    'typical_p': 1.0,
    'top_a': 0.52,
    #'epsion_cutoff': 1.49,
    #'eta_cutoff': 10.42,
    'length_penalty': 0.8,
}


def query_api(prompt, preset=None) -> str:
    url = f"{API_ENDPOINT}/v1/generate"

    if not preset:
        preset = DIVINE

    data = {**preset, "prompt": prompt, "truncation_length": 1024 * 16, "max_new_tokens": 1000}
    resp = requests.post(url, json=data)
    resp.raise_for_status()
    data = resp.json()["results"][0]["text"].strip()
    return data
