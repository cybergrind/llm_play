#!/usr/bin/env python3
import logging

from transformers import T5ForConditionalGeneration, T5Tokenizer

from data.texts import TEXT1


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger('summary_t5')


is_local = False
TOKENIZER = 'google/flan-t5-base'
# MODEL = 'google/flan-t5-base'
MODEL = 'google/t5-v1_1-large'


def t5_summarize(txt):
    tokenizer = T5Tokenizer.from_pretrained(TOKENIZER, local_files_only=is_local)
    model = T5ForConditionalGeneration.from_pretrained(
        MODEL, device_map='auto', local_files_only=is_local
    )
    prompt = 'summarize: ' + txt
    input_ids = tokenizer.encode(prompt, return_tensors='pt', max_length=1024).to('cuda')
    outputs = model.generate(input_ids, max_new_tokens=512, early_stopping=True)
    return tokenizer.decode(outputs[0])


def main():
    print(t5_summarize(TEXT1))


if __name__ == '__main__':
    main()
