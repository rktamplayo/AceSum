import argparse
import json
import rouge
import numpy as np


def rouge_preprocess(text):
  text = rouge.Rouge.REMOVE_CHAR_PATTERN.sub(' ', text.lower()).strip()
  tokens = rouge.Rouge.tokenize_text(rouge.Rouge.KEEP_CANNOT_IN_ONE_WORD.sub('_cannot_', text))
  rouge.Rouge.stem_tokens(tokens)
  preprocessed_text = rouge.Rouge.KEEP_CANNOT_IN_ONE_WORD_REVERSED.sub('cannot', ' '.join(tokens))
  return preprocessed_text


def calculate(gold_sums, pred_sums, num_aspects=6, rouge_eval=None):
  if rouge_eval is None:
    rouge_eval = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                             max_n=2,
                             limit_length=False,
                             apply_avg=True,
                             apply_best=False,
                             alpha=0.5, # Default F1_score
                             stemming=True)

  scores = rouge_eval.get_scores(pred_sums, gold_sums)

  rouge_l = scores['rouge-l']['f'] * 100
  rouge_1 = scores['rouge-1']['f'] * 100
  rouge_2 = scores['rouge-2']['f'] * 100

  return rouge_1, rouge_2, rouge_l
