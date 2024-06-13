#!/usr/bin/env python3

import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path


def predict(sentence, model, tokenizer):
    # Add [CLS] and [SEP] tokens
    sentence = "[CLS] " + sentence + " [SEP]"
    # Tokenize the sentence
    tokenized_text = tokenizer.tokenize(sentence)

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define a segment id (0 for all tokens; we don't have a second sequence)
    segments_ids = [0] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Make prediction
    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        logits = outputs.logits
        # You might want to convert logits to probabilities or extract the predicted label.
        predicted_label = torch.argmax(logits, dim=1).item()

    return predicted_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", "-m", type=Path)
    parser.add_argument("sentence", type=str)
    args = parser.parse_args()

    # Load the model
    model = BertForSequenceClassification.from_pretrained(args.model_dir)

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    model.eval()

    label = predict(args.sentence, model, tokenizer)
    print(label)


if __name__ == "__main__":
    main()

