#!/usr/bin/env python

from PIL import Image
from accelerate import Accelerator
from io import BytesIO
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
import requests


paths = [x for x in Path(".").glob("./*.txt")]

# Read the content from the files, ignoring or replacing invalid characters
file_contents = []
for path in paths:
    print(f"Reading from {path}...")
    file_contents.append(path.read_text())

# Join the contents into a single string
text = "\n".join(file_contents)

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train_from_iterator([text], vocab_size=52000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

token_dir = Path("./KantaiBERT")
token_dir.mkdir(parents=True, exist_ok=True)
files = tokenizer.save_model(str(token_dir))

print("Tokenizer model saved to:")
print()
for f in files:
    print(f"  * {f}")
