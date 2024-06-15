#!/usr/bin/env python

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

tokenizer = ByteLevelBPETokenizer(
    "./KantaiBERT/vocab.json",
    "./KantaiBERT/merges.txt",
)

text = "The Critique of Pure Reason."
print(f"source text: {text}")
v = tokenizer.encode(text)
print(f"tokenized: {v.tokens}")
print(v)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
print("Start and end tokens added.")

tokenizer.enable_truncation(max_length=512)
v = tokenizer.encode(text)
print(f"tokenized: {v.tokens}")
print(v)

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
