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

from transformers import RobertaConfig, RobertaTokenizer

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
print(config)

tokenizer = RobertaTokenizer.from_pretrained("./KantaiBERT", max_length=512)
print(tokenizer)

from transformers import RobertaForMaskedLM

print("Loading a Roberta model...")
model = RobertaForMaskedLM(config=config)
print(model)
print(f"number of parameters: {model.num_parameters()}")

LP = list(model.parameters())
lp = len(LP)
print(f"Model parameter length: {lp}")

import math

n_params = 0
for i, p in enumerate(LP):
    this_params = math.prod(p.shape)
    n_params += this_params
    print(f"shape of tensor {i}: {p.shape} (param count: {this_params})")

print(f"number of parameters: {n_params} (calculated)")

print("Building the dataset...")

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./kant.txt",
    block_size=128,
)

print("Defining a data collator...")

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

print("Initializing the trainer...")

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./KantaiBERT",
    overwrite_output_dir=True,
    num_train_epochs=2, #can be increased
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

print("Pre-train the model")

trainer.train()

print("Saving the final model (+tokenizer + config) to disk")
trainer.save_model("./KantaiBERT")
