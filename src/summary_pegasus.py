#!/usr/bin/env python3
import logging

import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

from data.texts import TEXT0


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger('summary_t5')


is_local = True
MODEL = 'google/pegasus-xsum'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def pegasus_summarize(txt):
    tokenizer = PegasusTokenizer.from_pretrained(MODEL, local_files_only=is_local)
    model = PegasusForConditionalGeneration.from_pretrained(MODEL, local_files_only=is_local).to(
        DEVICE
    )
    prompt = 'summarize: ' + txt
    input_ids = tokenizer(prompt, return_tensors='pt', truncation=True, padding='longest').to(
        DEVICE
    )
    outputs = model.generate(**input_ids, max_length=1024, early_stopping=True, min_length=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    print(pegasus_summarize(TEXT0))


if __name__ == '__main__':
    main()
