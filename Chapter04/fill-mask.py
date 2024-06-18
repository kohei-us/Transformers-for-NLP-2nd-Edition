#!/usr/bin/env python3

import argparse
from transformers import pipeline


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    fill_mask = pipeline(
        "fill-mask",
        model="./KantaiBERT",
        tokenizer="./KantaiBERT"
    )

    while True:
        v = input(">> ")
        print(v)
        results = fill_mask(v)
        for res in results:
            print(res)


if __name__ == "__main__":
    main()
