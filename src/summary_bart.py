#!/usr/bin/env python3
import logging

import torch
from transformers import BartTokenizer, BartForConditionalGeneration


from data.texts import TEXTS


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger('summary_t5')


is_local = True
MODEL = 'facebook/bart-large-cnn'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def bart_summarize(txt):
    tokenizer = BartTokenizer.from_pretrained(MODEL)
    model = BartForConditionalGeneration.from_pretrained(MODEL).to(DEVICE)

    input_ids = tokenizer.encode(txt, return_tensors='pt', truncation=True, padding='longest').to(
        DEVICE
    )
    outputs = model.generate(input_ids, max_length=1024, early_stopping=True, min_length=32)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    out = bart_summarize(TEXTS[1])
    print('='*80)
    print(out)


if __name__ == '__main__':
    main()
