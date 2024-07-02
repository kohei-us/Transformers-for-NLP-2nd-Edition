#!/usr/bin/env python3

import torch
import json

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

model = T5ForConditionalGeneration.from_pretrained("t5-large")
tokenizer = T5Tokenizer.from_pretrained("t5-large", legacy=True)

device = torch.device("cuda")

print(model.config)
print(model)
