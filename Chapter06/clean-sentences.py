#!/usr/bin/env python3

import argparse
import re
import string
import unicodedata
import pickle
from pathlib import Path


# prepare regex for char filtering
RE_PRINT = re.compile('[^%s]' % re.escape(string.printable))
# prepare translation table for removing punctuation
TABLE = str.maketrans('', '', string.punctuation)


def sentence_lengths(sentences):
	lengths = [len(s.split()) for s in sentences]
	return min(lengths), max(lengths)


def clean_lines(lines):

	cleaned = list()

	for line in lines:
		# normalize unicode characters
		line = unicodedata.normalize('NFD', line).encode('ascii', 'ignore')
		line = line.decode('UTF-8')
		# tokenize on white space
		line = line.split()
		# convert to lower case
		line = [word.lower() for word in line]
		# remove punctuation from each token
		line = [word.translate(TABLE) for word in line]
		# remove non-printable chars form each token
		line = [RE_PRINT.sub('', w) for w in line]
		# remove tokens with numbers in them
		line = [word for word in line if word.isalpha()]
		# store as string
		cleaned.append(' '.join(line))

	return cleaned


def preprocess(input: Path):
    sentences = input.read_text().strip().split('\n')
    print("* sample sentences")
    for i, s in enumerate(sentences[:2]):
        print(f"  - {i}: {s}")

    minlen, maxlen = sentence_lengths(sentences)
    print(f"* sentence length: min={minlen}; max={maxlen}")

    print("* cleaning sentences...")
    cleaned = clean_lines(sentences)
    print("* sample sentences (cleaned)")
    for i, s in enumerate(cleaned[:2]):
        print(f"  - {i}: {s}")

    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("en", type=Path)
    parser.add_argument("fr", type=Path)
    args = parser.parse_args()

    print("Preprocessing English...")
    sentences = preprocess(args.en)
    Path("sentences-en.pkl").write_bytes(pickle.dumps(sentences))
    print("Preprocessing French...")
    sentences = preprocess(args.fr)
    Path("sentences-fr.pkl").write_bytes(pickle.dumps(sentences))


if __name__ == "__main__":
    main()

