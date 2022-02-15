import argparse
import os
import random

import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize

from data_pipeline import aspect_detection_collate
from model import MIL

import time


def soft_margin(a, b):
  a = np.maximum(0, a)
  b = np.maximum(0, b)
  return np.log(1 + np.exp(-a * b)).sum()


def prepare_model(args):
  tokenizer = AutoTokenizer.from_pretrained(args.model_type)
  model = MIL(args)
  model.cuda()
  best_point = torch.load(args.load_model)
  model.load_state_dict(best_point['model'])
  model.eval()
  return model, tokenizer


def create_train_data(args,
                      min_token_frequency=5,
                      min_sentences=50,
                      min_reviews=10,
                      min_summary_tokens=30, #30, 60
                      max_summary_tokens=100, #60, 100
                      max_tokens=512,
                      num_keywords=10):
  # obtain data
  data = []

  f = open('data/' + args.dataset + '/train.jsonl', 'r')
  for line in tqdm(f):
    inst = json.loads(line.strip())
    data.append(inst)

  f.close()

  random.shuffle(data)

  # get model
  model, tokenizer = prepare_model(args)

  dataset_file = 'data/' + args.dataset + '/train.sum.jsonl'
  
  summary_set = set()
  if os.path.exists(dataset_file):
    # this is to make sure there are no duplicates when
    # script is ran twice due to an unexpected error.
    f = open(dataset_file, 'r')
    for line in f:
      try:
        inst = json.loads(line.strip())
      except:
        continue
      summary_set.add(inst['summary'])

    f.close()

  count = 0
  for inst in tqdm(data):
    total_reviews = inst['reviews']

    for i in range(0, len(total_reviews), 1000):
      reviews = total_reviews[i:i+1000]
      reviews = [review for review in reviews if len(review['sentences']) != 0]

      # remove instances with few reviews
      if len(reviews) < min_reviews:
        continue

      # tokenize reviews
      tok_reviews = []
      for review in reviews:
        tok_reviews.append([tokenizer.encode(sentence) for sentence in review['sentences']])

      sentence_switches_list = []
      word_switches_list = []
      document_switches = []

      # run model
      print('running model...')
      for j in range(0, len(tok_reviews), 2):
        tok_reviews_batch = tok_reviews[j:j+2]

        inp_batch = [(review, -1) for review in tok_reviews_batch]
        inp_batch, _ = aspect_detection_collate(inp_batch)
        inp_batch = inp_batch.cuda()

        with torch.no_grad():
          preds = model(inp_batch)
        document_pred = preds['document'].tolist()
        sentence_pred = preds['sentence'].cpu().detach().numpy()
        word_pred = preds['word'].cpu().detach().numpy()

        sentence_weight = preds['sentence_weight'].cpu().detach().numpy()
        word_weight = preds['word_weight'].cpu().detach().numpy()

        document_switches += document_pred

        for k, sentences in enumerate(tok_reviews_batch):
          sentence_switches = []
          word_switches = []
          for l, sentence in enumerate(sentences):
            tokens = tokenizer.convert_ids_to_tokens(sentence)[:100]

            sentence = tokenizer.decode(sentence, skip_special_tokens=True)
            pred = sentence_pred[k,l]
            weight = sentence_weight[k,l]
            sentence_switch = pred * weight
            sentence_switches.append((sentence, sentence_switch))

            word_switches_of_sentence = {}

            word = ""
            pred = []
            weight = []
            for m, token in enumerate(tokens):
              if token[0] == '\u0120':
                # start of a new token; reset values
                word = word.replace('<s>', '')
                token = token[1:]
                pred = np.max(pred, axis=0)
                weight = np.max(weight, axis=0)
                word_switch = np.maximum(0, pred * weight)
                if word not in word_switches_of_sentence:
                  word_switches_of_sentence[word] = 0
                word_switches_of_sentence[word] += word_switch

                word = ""
                pred = []
                weight = []

              word += token
              pred.append(word_pred[k,l,m])
              weight.append(word_weight[k,l,m])

            word_switches_of_sentence = [(word, word_switches_of_sentence[word]) for word in word_switches_of_sentence] # list of switches
            word_switches.append(word_switches_of_sentence) # list of list of switches

          sentence_switches_list.append(sentence_switches)
          word_switches_list.append(word_switches) # list of list of list of switches

      # sample summary and its reviews, keywords
      print('creating data...')
      for s_id, summary in enumerate(reviews):
        sentence_list = []
        for j, sentence in enumerate(summary['sentences']):
          sentence_switch = sentence_switches_list[s_id][j][1]
          contains_aspect = np.any([x>0 for x in sentence_switch])
          if not contains_aspect:
            continue
          sentence_list.append(sentence)
        summary = ' '.join(sentence_list)

        if summary in summary_set:
          continue

        # check min max length
        if len(summary.split()) < min_summary_tokens or len(summary.split()) > max_summary_tokens:
          continue

        document_switch = document_switches[s_id]

        # remove sentences of this review
        sentence_switches = []
        #word_switches = word_switches_list[s_id]
        word_switches = [] # list of list of switches
        for j in range(len(sentence_switches_list)):
          if j != s_id:
            sentence_switches += sentence_switches_list[j]
            word_switches += word_switches_list[j]

        # get sentences
        sentence_scores = [soft_margin(sentence_switch[-1], document_switch) for sentence_switch in sentence_switches]
        sentence_ids = np.argsort(sentence_scores)

        input_length = 0
        new_reviews = []
        for j, idx in enumerate(sentence_ids):
          if sentence_scores[idx] == 1e9:
            break
          if input_length > 600:
            break
          try:
            sentence = sentence_switches[idx][0]
          except:
            continue
          input_length += len(sentence.split())
          new_reviews.append(sentence)

        sentence_ids = sentence_ids[:j]
        if len(sentence_ids) == 0: # no related sentences
          continue

        # combine word switches
        word_switches_dict = {}
        for idx in range(len(word_switches)):
          if idx not in sentence_ids:
            continue
          for word, switch in word_switches[idx]:
            if word not in word_switches_dict:
              word_switches_dict[word] = np.zeros(args.num_aspects)
            word_switches_dict[word] += np.maximum(0, switch)

        word_switches_final = [(word, word_switches_dict[word]) for word in word_switches_dict]
              
        word_scores = [soft_margin(word_switch[-1], document_switch) for word_switch in word_switches_final]
        word_switches = [
          (word_switch, word_score) 
          for word_switch, word_score in zip(word_switches_final, word_scores) 
          if word_score != 1e9
        ]
        word_switches = sorted(word_switches, key=lambda a: a[-1])[:num_keywords]
        keywords = [word_switch[0][0] for word_switch in word_switches]

        pair = {}
        pair['summary'] = summary
        pair['reviews'] = new_reviews
        pair['keywords'] = keywords
        pair['switch'] = document_switch
        f = open(dataset_file, 'a')
        f.write(json.dumps(pair) + '\n')
        f.close()
        count += 1

      print(count)


def create_aspect_test_data(args,
                            num_keywords=10):
  # get model
  model, tokenizer = prepare_model(args)

  # prepare switch map
  switch_map = {}
  keyword_dirs = os.listdir('seeds/' + args.dataset)
  keyword_dirs = sorted(keyword_dirs)
  for i, file in enumerate(keyword_dirs):
    aspect = file[:-4]
    switch = [0] * len(keyword_dirs)
    switch[i] = 1
    switch_map[aspect] = switch

  for split in ['dev', 'test']:
    # obtain data
    f = open('data/' + args.dataset + '/' + split + '.json', 'r')
    data = json.load(f)
    f.close()

    f = open('data/' + args.dataset + '/' + split + '.sum.aspect.jsonl', 'w')

    for inst in tqdm(data):
      reviews = inst['reviews']

      # tokenize reviews
      tok_reviews = []
      for review in reviews:
        tok_reviews.append([tokenizer.encode(sentence.lower()) for sentence in review['sentences']])

      sentence_switches = []
      word_switches = {}

      # run model
      print('running model...')
      for j in range(0, len(tok_reviews), 2):
        tok_reviews_batch = tok_reviews[j:j+2]

        inp_batch = [(review, -1) for review in tok_reviews_batch]
        inp_batch, _ = aspect_detection_collate(inp_batch)
        inp_batch = inp_batch.cuda()

        with torch.no_grad():
          preds = model(inp_batch)
        sentence_pred = preds['sentence'].cpu().detach().numpy()
        word_pred = preds['word'].cpu().detach().numpy()

        sentence_weight = preds['sentence_weight'].cpu().detach().numpy()
        word_weight = preds['word_weight'].cpu().detach().numpy()

        for k, sentences in enumerate(tok_reviews_batch):
          for l, sentence in enumerate(sentences):
            tokens = tokenizer.convert_ids_to_tokens(sentence)[:100]

            sentence = tokenizer.decode(sentence, skip_special_tokens=True)
            pred = sentence_pred[k,l]
            weight = sentence_weight[k,l]
            sentence_switch = pred * weight
            sentence_switches.append((sentence, sentence_switch))

            word = ""
            pred = []
            weight = []
            for m, token in enumerate(tokens):
              if token[0] == '\u0120':
                # start of a new token; reset values
                word = word.replace('<s>', '')
                token = token[1:]
                pred = np.max(pred, axis=0)
                weight = np.max(weight, axis=0)
                word_switch = pred * weight
                if word not in word_switches:
                  word_switches[word] = 0
                word_switches[word] += word_switch

                word = ""
                pred = []
                weight = []

              word += token
              pred.append(word_pred[k,l,m])
              weight.append(word_weight[k,l,m])

      word_switches = [(word, word_switches[word]) for word in word_switches]

      for aspect in switch_map:
        document_switch = switch_map[aspect]

        random.shuffle(word_switches)
        random.shuffle(sentence_switches)

        # get keywords
        word_scores = [soft_margin(word_switch[-1], document_switch) for word_switch in word_switches]
        word_switches = [
          (word_switch, word_score) 
          for word_switch, word_score in zip(word_switches, word_scores) 
          if word_score != 1e9
        ]
        word_switches = sorted(word_switches, key=lambda a: a[-1])[:num_keywords]
        keywords = [word_switch[0][0] for word_switch in word_switches]

        # get sentences
        sentence_scores = [soft_margin(sentence_switch[-1], document_switch) for sentence_switch in sentence_switches]
        sentence_switches = [
          (sentence_switch, sentence_score) 
          for sentence_switch, sentence_score in zip(sentence_switches, sentence_scores) 
          #if sentence_score != 1e9
        ]
        sentence_switches = sorted(sentence_switches, key=lambda a: a[-1])

        input_length = 0
        idx = 0
        new_reviews = []
        for idx in range(len(sentence_switches)):
          if input_length > 600:
            break
          try:
            sentence = sentence_switches[idx][0][0]
          except:
            continue
          input_length += len(sentence.split())
          new_reviews.append(sentence)

        pair = {}
        pair['summary'] = [x.lower() for x in inst['summaries'][aspect]]
        pair['reviews'] = new_reviews
        pair['keywords'] = keywords
        pair['switch'] = document_switch
        f.write(json.dumps(pair) + '\n')

    f.close()


def create_general_test_data(args,
                             num_keywords=10):
  # get model
  model, tokenizer = prepare_model(args)
  model.cuda()
  model.eval()

  for split in ['dev', 'test']:
    # obtain data
    data = []

    f = open('data/' + args.dataset + '/' + split + '.json', 'r')
    data = json.load(f)
    f.close()

    f = open('data/' + args.dataset + '/' + split + '.sum.general.jsonl', 'w')

    for inst in tqdm(data):
      reviews = inst['reviews']

      # tokenize reviews
      tok_reviews = []
      for review in reviews:
        sentences = [' '.join(word_tokenize(sentence.lower())) for sentence in review['sentences']]
        tok_reviews.append([tokenizer.encode(sentence) for sentence in sentences])

      sentence_switches = []
      word_switches = {}

      # run model
      print('running model...')
      for j in range(0, len(tok_reviews), 2):
        tok_reviews_batch = tok_reviews[j:j+2]

        inp_batch = [(review, -1) for review in tok_reviews_batch]
        inp_batch, _ = aspect_detection_collate(inp_batch)
        inp_batch = inp_batch.cuda()

        with torch.no_grad():
          preds = model(inp_batch)
        sentence_pred = preds['sentence'].cpu().detach().numpy()
        word_pred = preds['word'].cpu().detach().numpy()

        sentence_weight = preds['sentence_weight'].cpu().detach().numpy()
        word_weight = preds['word_weight'].cpu().detach().numpy()

        for k, sentences in enumerate(tok_reviews_batch):
          for l, sentence in enumerate(sentences):
            tokens = tokenizer.convert_ids_to_tokens(sentence)[:100]

            sentence = tokenizer.decode(sentence, skip_special_tokens=True)
            pred = sentence_pred[k,l]
            weight = sentence_weight[k,l]
            sentence_switch = pred * weight
            sentence_switches.append((sentence, sentence_switch))

            word = ""
            pred = []
            weight = []
            for m, token in enumerate(tokens):
              if token[0] == '\u0120':
                # start of a new token; reset values
                word = word.replace('<s>', '')
                token = token[1:]
                pred = np.max(pred, axis=0)
                weight = np.max(weight, axis=0)
                word_switch = pred * weight
                if word not in word_switches:
                  word_switches[word] = 0
                word_switches[word] += word_switch

                word = ""
                pred = []
                weight = []

              word += token
              pred.append(word_pred[k,l,m])
              weight.append(word_weight[k,l,m])

      word_switches = [(word, word_switches[word]) for word in word_switches]

      document_switch = [1] * args.num_aspects

      random.shuffle(word_switches)
      random.shuffle(sentence_switches)

      # get keywords
      word_scores = [soft_margin(word_switch[-1], document_switch) for word_switch in word_switches]
      word_switches = [
        (word_switch, word_score) 
        for word_switch, word_score in zip(word_switches, word_scores) 
        if word_score != 1e9
      ]
      word_switches = sorted(word_switches, key=lambda a: a[-1])[:num_keywords]
      keywords = [word_switch[0][0] for word_switch in word_switches]

      # get sentences
      sentence_scores = [soft_margin(sentence_switch[-1], document_switch) for sentence_switch in sentence_switches]
      sentence_switches = [
        (sentence_switch, sentence_score) 
        for sentence_switch, sentence_score in zip(sentence_switches, sentence_scores) 
        if sentence_score != 1e9
      ]
      sentence_switches = sorted(sentence_switches, key=lambda a: a[-1])

      input_length = 0
      idx = 0
      new_reviews = []
      for idx in range(len(sentence_switches)):
        if sentence_switches[idx][1] == 1e9:
          break
        if input_length > 600:
          break
        try:
          sentence = sentence_switches[idx][0][0]
        except:
          continue
        input_length += len(sentence.split())
        new_reviews.append(sentence)

      pair = {}
      pair['summary'] = [x.lower() for x in inst['summaries']['general']]
      pair['reviews'] = new_reviews
      pair['keywords'] = keywords
      pair['switch'] = document_switch
      f.write(json.dumps(pair) + '\n')

    f.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('-mode', default='train', type=str)

  parser.add_argument('-dataset', default='space', type=str)
  parser.add_argument('-num_aspects', default=6, type=int)

  parser.add_argument('-load_model', default=None, type=str)
  parser.add_argument('-model_type', default='distilroberta-base', type=str)

  args = parser.parse_args()

  if args.mode == 'train':
    create_train_data(args)
  elif args.mode == 'eval-aspect':
    create_aspect_test_data(args)
  elif args.mode == 'eval-general':
    create_general_test_data(args)