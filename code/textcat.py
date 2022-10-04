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
        "lm1",
        type=Path,
        help="path to the trained model for the genuine",
    )
    parser.add_argument(
        "lm2",
        type=Path,
        help="path to the trained model for the class one (spam)",
    )
    parser.add_argument(
        "prior_probability",
        type=float,
        help="prior probability for the class zero (gen)",
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0
    x: Wordtype;
    y: Wordtype;
    z: Wordtype  # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)
    return log_prob


def find_last(text, pattern):
    return text.rfind(pattern)


def find_second_last(text, pattern):
    return text.rfind(pattern, 0, text.rfind(pattern))


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)

    log.info("Testing...")
    lm1 = LanguageModel.load(args.lm1)
    lm2 = LanguageModel.load(args.lm2)

    assert lm1.vocab == lm2.vocab

    prior_probability = args.prior_probability

    lm1count = 0
    lm2count = 0
    ct_incorrect_classified = 0
    for file in args.test_files:
        dir_name = str(file.parents[0])
        true_class = str(dir_name[str(dir_name).rindex('\\') + 1:])
        log_prob1: float = file_log_prob(file, lm1)
        log_prob2: float = file_log_prob(file, lm2)
        posteriori1 = log_prob1 + math.log(prior_probability)
        posteriori2 = log_prob2 + math.log(1 - prior_probability)
        if posteriori1 >= posteriori2:
            predicted_class = 'gen'
            print(f"{args.lm1}\t{file}")
            lm1count += 1
        else:
            predicted_class = 'spam'
            print(f"{args.lm2}\t{file}")
            lm2count += 1
        if predicted_class != true_class:
            ct_incorrect_classified += 1
    lm1prob = lm1count / (lm1count + lm2count)
    lm2prob = lm2count / (lm1count + lm2count)
    print(f"{lm1count} files were more probably {args.lm1} ({lm1prob})")
    print(f"{lm2count} files were more probably {args.lm2} ({lm2prob})")
    # print(f"Total error rate {ct_incorrect_classified / len(args.test_files) * 100}")


if __name__ == "__main__":
    main()
