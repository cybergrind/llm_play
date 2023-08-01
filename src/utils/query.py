import requests


API_ENDPOINT = "http://localhost:5000/api"


def make_airoboros(text):
    return f"""
Context: A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input.
Continue the chat dialogue below. Write a single reply for the character "ASSISTANT".

USER: {text}
ASSISTANT:
"""


def query_api(prompt, temperature=0.7):
    url = f"{API_ENDPOINT}/v1/generate"

    data = {
        "prompt": prompt,
        "truncation_length": 1024 * 16,
        "max_new_tokens": 1000,
        "temperature": temperature,
    }
    resp = requests.post(url, json=data)
    resp.raise_for_status()
    data = resp.json()["results"][0]["text"].strip()
    return data
