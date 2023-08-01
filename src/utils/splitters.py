#!/usr/bin/env python3
import logging
import sys
from pathlib import Path
from typing import Callable

from nltk.tokenize import TextTilingTokenizer

from utils.query import query_api


log = logging.getLogger("splitters")
MIN_TEXT_SIZE = 2048  # do not split texts under this size
SPLIT_PROMPT = '''
Context: A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input.

Continue the chat dialogue below. Write a single reply for the character "ASSISTANT".

USER:

<TEXT_START>
{text}
<TEXT_END>

Is this text has more then one chapter?
Answer only Y or N
ASSISTANT:
'''


def split_into_chapters_llm(text: str, llm_func: Callable[str, str]):
    """
    Use `llm_func` to split text into chapters
    """
    while len(text) < MIN_TEXT_SIZE:
        yield text

    split_prompt = SPLIT_PROMPT.format(text=text)
    split = llm_func(split_prompt, temperature=0.0001)
    log.info(f"Split: {split}")


def split_into_chapters_ttt(text: str):
    ttt = TextTilingTokenizer(w=40, k=100)
    splits = ttt.tokenize(text)
    return splits


def main():
    fname = sys.argv[1]
    log.info(f"Reading {fname}")
    text = Path(fname).read_text()
    log.info(f"Text size: {len(text)}")
    # resp = split_into_chapters_llm(text, query_api)
    # next(resp)

    splits = list(split_into_chapters_ttt(text))
    log.info(f"Splits: {len(splits)=}")
    for i, split in enumerate(splits):
        out_fname = f"{fname}.{i}"
        log.info('=' * 80)
        log.info(f"Writing {out_fname} => {len(split)=} => {split}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
