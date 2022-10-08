#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.  
"""
import re
import math
import argparse
from pathlib import Path
from operator import itemgetter
from probs import Wordtype, LanguageModel, read_trigrams_from_sentence

import logging

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="path to the trained model",
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


def file_total_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    with open(file) as f:
        correct_length = int(f.readline().split()[0])
        # transcriptions = {}
        err_rates = []
        u_probs = []
        transcript_lengths = []
        strings = []

        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip("\n")
            err_rate, u_prob, transcript_len, sentence = re.split("\t", line)
            err_rates.append(float(err_rate))
            u_probs.append(float(u_prob))
            transcript_lengths.append(int(transcript_len))
            strings.append(sentence)

        log_probs = []

        for sentence in strings:
            x: Wordtype;
            y: Wordtype;
            z: Wordtype
            log_prob = 0
            for (x, y, z) in read_trigrams_from_sentence(sentence, lm.vocab):
                log_prob += lm.log_prob(x, y, z)  # log p(z | xy)
            log_probs.append(log_prob)

        total_probs = [sum(x) for x in zip(u_probs, log_probs)]
        chosen_transcription_index, _ = max(enumerate(total_probs), key=itemgetter(1))
        chosen_transcript_err_rate = err_rates[chosen_transcription_index]
        chosen_transcript_length = transcript_lengths[chosen_transcription_index]
        return chosen_transcript_err_rate, chosen_transcript_length, correct_length


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)

    log.info("Testing...")
    lm = LanguageModel.load(args.model)

    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # We'll print that first.

    log.info("Per-file log-probabilities:")
    total_log_prob = 0.0
    total_lengths = 0
    total_err_counts = 0
    for file in args.test_files:
        err_rate, transcript_length, correct_length = file_total_prob(file, lm)
        print(f"{err_rate:.3f}\t{file}")
        total_lengths += transcript_length
        total_err_counts += err_rate * correct_length
    overall_err_rate = total_err_counts / total_lengths
    print(f"{overall_err_rate:.3f}\t OVERALL")


if __name__ == "__main__":
    main()
