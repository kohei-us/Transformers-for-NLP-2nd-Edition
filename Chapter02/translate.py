#!/usr/bin/env python

from transformers import pipeline
translator = pipeline("translation_en_to_fr")
print(translator("It is easy to translate languages with transformers", max_length=40))
