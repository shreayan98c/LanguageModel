#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.
"""
import argparse
import logging
import math
from pathlib import Path

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="path to the trained model",
    )
    parser.add_argument(
        "num_sentences",
        type=int,
        help=f"Number of sentences to be generated",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        help="Maximum length of the sentences to be shown",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    # args, unknown = parser.parse_known_args()

    logging.basicConfig()

    log.info("Testing...")
    lm = LanguageModel.load(args.model)
    num_sentences = args.num_sentences
    max_length = args.max_length

    for _ in range(num_sentences):
        sentence = lm.sample(max_length)
        print(sentence)


if __name__ == "__main__":
    main()
