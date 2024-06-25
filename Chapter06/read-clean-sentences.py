#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path
from collections import Counter


# create a frequency table for all words
def to_vocab(lines):
	vocab = Counter()
	for line in lines:
		tokens = line.split()
		vocab.update(tokens)
	return vocab


# remove all words with a frequency below a threshold
def trim_vocab(vocab, min_occurance):
	tokens = [k for k, c in vocab.items() if c >= min_occurance]
	return set(tokens)


# mark all OOV with "unk" for all lines
def update_dataset(lines, vocab):
	new_lines = list()
	for line in lines:
		new_tokens = list()
		for token in line.split():
			if token in vocab:
				new_tokens.append(token)
			else:
				new_tokens.append('unk')
		new_line = ' '.join(new_tokens)
		new_lines.append(new_line)

	return new_lines


def process_language(name, inpath, outpath):
    lines = pickle.loads(inpath.read_bytes())
    vocab = to_vocab(lines)
    print(f"{name} vocabulary: {len(vocab)}")
    vocab = trim_vocab(vocab, 5)
    print(f"{name} vocabulary (trimmed): {len(vocab)}")
    lines = update_dataset(lines, vocab)

    outpath.write_bytes(pickle.dumps(lines))

    for i, line in enumerate(lines[:8]):
        print(f"- {i}: {line}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("en", type=Path)
    parser.add_argument("fr", type=Path)
    args = parser.parse_args()

    process_language("English", args.en, Path("./vocab-en.pkl"))
    process_language("French", args.fr, Path("./vocab-fr.pkl"))


if __name__ == "__main__":
    main()

