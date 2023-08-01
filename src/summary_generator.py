#!/usr/bin/env python3
"""
summarize text using text-generation-webui API
"""
import argparse
import json
import logging
from pathlib import Path

from utils.query import query_api, make_airoboros
from utils.splitters import split_into_chapters_ttt as split


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
    return parser.parse_args()



BIG_SUMMARY_START = "<STORY_START>\n"
BIG_SUMMARY_END = """
<STORY_END>
======================================================================
SUMMARIZED STORY:

"""

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


def do_summary(text, args):
    prompt = f"{BIG_SUMMARY_START}\n{text}\n{BIG_SUMMARY_END}"
    data = query_api(make_airoboros(prompt))
    if not data:
        log.info("No data")
        return False
    return data


def do_big_summary(args):
    summaries = []
    for paragraph in split(args.input):
        data = None
        while not data:
            data = do_summary(paragraph, args)
            if not data:
                continue
            summaries.append(data)

    joined_summaries = '\n======\n'.join(summaries)
    with open('tmp.ttt', 'a') as f:
        f.write('Iteration:\n\n')
        f.write(joined_summaries)

    final_summary = do_summary(joined_summaries, args)
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
    content = "==============\n".join(summaries)
    prompt = f"{SUPER_SUMMARY_START}\n{content}\n{SUPER_SUMMARY_END}"
    data = query_api(prompt)
    args.output.write_text(data)


def main(args):
    curr_summaries = num_summaries(args.temp)
    while curr_summaries < args.num_interations:
        if do_big_summary(args):
            curr_summaries += 1
            log.debug(f"Iteration finished: {curr_summaries}")
    summary_of_summaries(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
