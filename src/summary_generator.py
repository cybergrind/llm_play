#!/usr/bin/env python3
"""
summarize text using text-generation-webui API
"""
import argparse
import json
import logging
from functools import partial
from pathlib import Path
from typing import Callable, Union

from utils.iterators import split_with_overlap
from utils.query import make_airoboros, query_api
from utils.splitters import split_into_chapters_ttt as split_chapters


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("summary_generator")


def parse_args():
    parser = argparse.ArgumentParser(description="DESCRIPTION")
    parser.add_argument("-t", "--temp", default="temporary.jsonl", type=Path)
    parser.add_argument("-n", "--num-interations", default=20, type=int)
    parser.add_argument(
        "input", help="input_file for summarization", type=lambda x: Path(x).read_text()
    )
    parser.add_argument("-o", "--output", default="out.txt", type=Path)
    parser.add_argument('--ctx', default=4096, type=int)
    parser.add_argument('--strategy', default='dichotomy', choices=['dichotomy', 'cummulative'])
    return parser.parse_args()


SEP = '\n====\n'
BIG_SUMMARY_START = (
    "Please summarize the following story in roughly 30 words, maintaining only the key points:\n"
)
BIG_SUMMARY_END = ''

SUPER_SUMMARY_START = "<SUMMARIES_START>\n"
SUPER_SUMMARY_END = """<SUMMARIES_END>
INSTRUCTION:
* given multiple summaries write final summary
* some summaries may be bad, don't use them in the final text
* some summaries may be good, use them in the final text
* some summaries may be good, but not about the topic, don't use them in the final text
* at least several summaries should prove that the text is about the topic
* final summary should be around 500-1000 characters

Write final summary below.
"""


class Summaries(list):
    def __init__(self, path, wipe=False):
        super().__init__()
        self.path = path
        if wipe and self.path.exists():
            self.path.unlink()
        self.load()

    def append(self, summary):
        super().append(summary)
        self.dump()

    def load(self):
        if not self.path.exists():
            return

        with self.path.open('r') as f:
            for line in f.readlines():
                data = json.loads(line)
                self.append(data['summary'])

    def dump(self):
        with self.path.open('w') as f:
            for summary in self:
                data = json.dumps({"summary": summary})
                f.write(data + '\n')


def summary_query(lst_or_text: Union[list, str], end=' '):
    if isinstance(lst_or_text, list):
        text = SEP.join(lst_or_text)
    else:
        text = lst_or_text
    prompt = f"{BIG_SUMMARY_START}{text}{BIG_SUMMARY_END}"
    return make_airoboros(prompt, end=end)


def do_summary(prompt, args):
    assert len(prompt) < args.ctx, f'{len(prompt)=} / {prompt=}'
    data = query_api(prompt)
    if not data:
        log.info("No data")
        return False
    return data


def check_size(ctx_size, text, func=None):
    if func:
        size = len(func(text))
    else:
        size = len(text)
    print(f'{size=}')
    return size < ctx_size


def recursive_summary(prepare: Callable[[list], str], summaries, args, nested=0):
    query = prepare(summaries)
    log.info(f'Recursive summary [{nested}]: {len(summaries)=} vs {len(query)=} vs {args.ctx=}')

    if len(query) > args.ctx:
        if args.strategy == 'dichotomy':
            func = partial(check_size, args.ctx, func=prepare)
            batched = split_with_overlap(summaries, overlap=0, func=func)
            subsummaries = Summaries(Path(f'recursive.{nested}.jsonl'), wipe=True)
            for batch in batched:
                data = ''
                not_summarized = True
                while not_summarized:
                    data = do_summary(prepare(batch), args)
                    assert data, f'No data? {data=}'
                    if len(data) > args.ctx // 2:
                        log.debug(f'retry summarization: {len(data)=}')
                        batch = [data]
                    else:
                        not_summarized = False
                subsummaries.append(data)
            return recursive_summary(prepare, subsummaries, args, nested=nested + 1)
        else:
            raise NotImplementedError
    return do_summary(query, args)


BIG_SUMMARIES = Path('tmp_ttt.jsonl')

def do_big_summary(args):
    summaries = Summaries(BIG_SUMMARIES)
    chapters_to_process = split_chapters(args.input)[len(summaries) :]
    for paragraph in chapters_to_process:
        data = None
        while not data:
            data = do_summary(summary_query(paragraph), args)
            if not data:
                continue
            summaries.append(data)
    final_summary = recursive_summary(
        partial(summary_query, end=" Summary: \n"), summaries, args
    )
    json_line = json.dumps({"summary": final_summary})

    # append to args.temp
    if not args.temp.exists():
        args.temp.touch()
    tmp: Path = args.temp
    with tmp.open("a") as f:
        f.write(json_line + "\n")
    return True


def num_summaries(path: Path):
    """
    return num of lines
    """
    if not path.exists():
        return 0
    return len(path.read_text().splitlines())


def summary_of_summaries(args):
    """
    run summarization on args.temp
    """
    summaries = []
    for line in args.temp.read_text().splitlines():
        if not line:
            continue
        data = json.loads(line)
        if not data or not data["summary"]:
            continue
        summaries.append(data["summary"])
    content = SEP.join(summaries)
    prompt = f"{SUPER_SUMMARY_START}\n{content}\n{SUPER_SUMMARY_END}"
    data = query_api(prompt)
    args.output.write_text(data)


def main(args):
    curr_summaries = num_summaries(args.temp)
    while curr_summaries < args.num_interations:
        if do_big_summary(args):
            curr_summaries += 1
            BIG_SUMMARIES.unlink()
            log.debug(f"Iteration finished: {curr_summaries}")
    summary_of_summaries(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
