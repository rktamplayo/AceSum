import argparse

import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM as Model
from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup

import os
from tqdm import tqdm

from calculate_rouge import calculate
from data_pipeline import SummarizationDataset


def train(args):
  print(args)

  print('Preparing data...')
  
  if args.train_file is None:
    if args.dataset == 'oposum':
      train_file = []
      for domain in ['bag', 'boots', 'bt', 'keyboard', 'tv', 'vacuum']:
        train_file.append(args.data_dir + '/' + args.dataset + '/' + domain + '/train.sum.jsonl')
    else:
      train_file = args.data_dir + '/' + args.dataset + '/train.sum.jsonl'
  else:
    train_file = args.train_file
  dataset = SummarizationDataset(
    train_file, 
    use_keywords=args.use_keywords, use_switch=args.use_switch)
  dataloader = DataLoader(dataset, batch_size=args.batch_size)

  if args.asp_dev_file is None:
    asp_dev_file = args.data_dir + '/' + args.dataset + '/dev.sum.aspect.jsonl' % args.seed_type
  else:
    asp_dev_file = args.asp_dev_file
  asp_dev_dataset = SummarizationDataset(
    asp_test_file, 
    use_keywords=args.use_keywords, use_switch=args.use_switch, shuffle=False)
  asp_dev_dataloader = DataLoader(asp_dev_dataset, batch_size=args.batch_size)
  f = open(asp_dev_file, 'r')
  lines = f.readlines()
  data = [json.loads(line) for line in lines]
  f.close()
  asp_gold_sums = [[summary.lower() for summary in inst['summary']] for inst in data]

  if args.gen_dev_file is None:
    gen_dev_file = args.data_dir + '/' + args.dataset + '/dev.sum.general.jsonl' % args.seed_type
  else:
    gen_dev_file = args.gen_dev_file
  gen_dev_dataset = SummarizationDataset(
    gen_dev_file,
    use_keywords=args.use_keywords, use_switch=args.use_switch, shuffle=False)
  gen_dev_dataloader = DataLoader(gen_dev_dataset, batch_size=args.batch_size)
  f = open(gen_dev_file, 'r')
  lines = f.readlines()
  data = [json.loads(line) for line in lines]
  f.close()
  gen_gold_sums = [[summary.lower() for summary in inst['summary']] for inst in data]

  print('Initializing model...')

  tokenizer = AutoTokenizer.from_pretrained(args.model_type)
  special_tokens = ['<rev>', '<key>', '<sum>', '<switch>']
  if args.use_switch != 'none':
    for i in range(args.num_aspects):
      special_tokens.append('<pos_%d>' % i)

  tokenizer.add_special_tokens(
    {'additional_special_tokens': special_tokens}
  )
  pad_id = tokenizer.pad_token_id

  model = Model.from_pretrained(args.model_type, return_dict=True)
  model.resize_token_embeddings(len(tokenizer))
  model.cuda()

  optimizer = AdamW(model.parameters(), lr=args.learning_rate)
  scheduler = get_cosine_schedule_with_warmup(
    optimizer, args.no_warmup_steps, args.no_train_steps)

  step = 0
  best_rouge = 0
  rng = np.random.default_rng()
  if args.load_model is not None:
    print('Loading model...')
    best_point = torch.load(args.load_model)
    model.load_state_dict(best_point['model'])
    optimizer.load_state_dict(best_point['optimizer'])
    scheduler.load_state_dict(best_point['scheduler'])
    step = best_point['step']

  print('Start training...')
  while step < args.no_train_steps:
    losses = []
    for _, (inp_batch, out_batch, switch_batch) in enumerate(tqdm(dataloader)):
      model.train()

      batch_encoding = tokenizer.prepare_seq2seq_batch(
        src_texts=inp_batch,
        tgt_texts=out_batch,
        max_length=args.max_length,
        max_target_length=args.max_target_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
      )

      inp_ids = batch_encoding['input_ids'].cuda()
      inp_mask = batch_encoding['attention_mask'].cuda()
      out_ids = batch_encoding['labels'].cuda()
      out_mask = torch.where(out_ids==0, 0, 1).unsqueeze(-1) # batch_size, out_len
      out_ids[out_ids==0] = -100
      dec_inp_ids = model._shift_right(out_ids)

      model_outputs = model(input_ids=inp_ids,
                            attention_mask=inp_mask,
                            decoder_input_ids=dec_inp_ids,
                            labels=out_ids,
                            output_hidden_states=True)

      loss = model_outputs.loss
      losses.append(loss.item())

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      scheduler.step()

      step += 1
      if step % args.check_every == 0:
        print('Step %d Loss %.4f' % (step, np.mean(losses)))
        model.eval()

        rouge_scores = []

        # generate general summaries
        gen_pred_sums = []
        for _, (inp_batch, out_batch, _) in enumerate(tqdm(gen_dev_dataloader)):
          batch_encoding = tokenizer.prepare_seq2seq_batch(
            src_texts=inp_batch,
            tgt_texts=out_batch,
            max_length=args.max_length,
            max_target_length=args.max_target_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
          )

          inp_ids = batch_encoding['input_ids'].cuda()

          preds = model.generate(
            inp_ids,
            min_length=60,
            max_length=args.max_target_length*2,
            num_beams=2,
            no_repeat_ngram_size=2,
            decoder_start_token_id=0,
            repetition_penalty=1,
            length_penalty=1,
          )

          for pred in preds:
            gen_pred_sums.append(tokenizer.decode(pred))

        gen_scores = calculate(gen_gold_sums, gen_pred_sums)
        rouge_scores += list(gen_scores)

        asp_pred_sums = []
        for _, (inp_batch, out_batch, _) in enumerate(tqdm(asp_dev_dataloader)):
          model.eval()

          batch_encoding = tokenizer.prepare_seq2seq_batch(
            src_texts=inp_batch,
            tgt_texts=out_batch,
            max_length=args.max_length,
            max_target_length=args.max_target_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
          )

          inp_ids = batch_encoding['input_ids'].cuda()

          preds = model.generate(
            inp_ids,
            min_length=10,
            max_length=args.max_target_length*2,
            num_beams=2,
            no_repeat_ngram_size=2,
            decoder_start_token_id=0,
            repetition_penalty=1,
            length_penalty=1,
          )

          for pred in preds:
            asp_pred_sums.append(tokenizer.decode(pred))

        asp_scores = calculate(asp_gold_sums, asp_pred_sums)
        rouge_scores += list(asp_scores)

        rouge = np.power(np.product(rouge_scores), 1.0/len(rouge_scores))

        print("ROUGE: %.4f" % rouge)
        if args.dataset == 'space':
          print("General Gold:", gen_gold_sums[0])
          print("General Pred:", gen_pred_sums[0])
        print("Aspect Gold:", asp_gold_sums[0])
        print("Aspect Pred:", asp_pred_sums[0])

        if rouge > best_rouge:
          print('Saving...')
          best_rouge = rouge
          torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'step': step,
            'loss': np.mean(losses)
          }, args.model_dir + '/' + args.dataset + '/' + args.model_name + '.best.%d.%.3f' % (step, rouge))

      if step % args.ckpt_every == 0:
        print('Saving...')
        torch.save({
          'model': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'scheduler': scheduler.state_dict(),
          'step': step,
          'loss': np.mean(losses)
        }, args.model_dir + '/' + args.dataset + '/' + args.model_name + '.%d.%.2f' % (step, np.mean(losses)))
        losses = []

      if step == args.no_train_steps:
        break


def evaluate(args, test_type='general'):
  print(args)

  print('Preparing data...')

  if (args.gen_test_file is None and test_type == 'general') or (args.asp_test_file is None and test_type == 'aspect'):
    test_file = args.data_dir + '/' + args.dataset + '/test.sum.%s.jsonl' % (args.seed_type, test_type)
  elif test_type == 'general':
    test_file = args.gen_test_file
  elif test_type == 'aspect':
    test_file = args.asp_test_file
  else:
    test_file = args.gen_test_file
  dataset = SummarizationDataset(
    test_file,
    use_keywords=args.use_keywords, use_switch=args.use_switch, shuffle=False)
  dataloader = DataLoader(dataset, batch_size=args.batch_size)

  print('Initializing model...')
  tokenizer = AutoTokenizer.from_pretrained(args.model_type)
  special_tokens = ['<rev>', '<key>', '<sum>', '<switch>']
  if args.use_switch != 'none':
    for i in range(args.num_aspects):
      special_tokens.append('<pos_%d>' % i)

  tokenizer.add_special_tokens(
    {'additional_special_tokens': special_tokens}
  )
  pad_id = tokenizer.pad_token_id

  model = Model.from_pretrained(args.model_type, return_dict=True)
  model.resize_token_embeddings(len(tokenizer))
  model.cuda()

  optimizer = AdamW(model.parameters(), lr=1e-5)
  scheduler = get_cosine_schedule_with_warmup(
    optimizer, args.no_warmup_steps, args.no_train_steps)

  assert args.load_model is not None
  best_point = torch.load(args.load_model)
  model.load_state_dict(best_point['model'])
  optimizer.load_state_dict(best_point['optimizer'])
  scheduler.load_state_dict(best_point['scheduler'])

  step = 0
  rng = np.random.default_rng()

  os.makedirs('output/' + args.dataset, exist_ok=True)
  f = open('output/' + args.dataset + '/' + args.load_model.split('/')[-1] + '.out.' + test_type, 'w')
  for _, (inp_batch, out_batch, _) in enumerate(tqdm(dataloader)):
    model.eval()

    batch_encoding = tokenizer.prepare_seq2seq_batch(
      src_texts=inp_batch,
      tgt_texts=out_batch,
      max_length=args.max_length,
      max_target_length=args.max_target_length,
      padding=True,
      truncation=True,
      return_tensors='pt'
    )

    inp_ids = batch_encoding['input_ids'].cuda()
    inp_mask = batch_encoding['attention_mask'].cuda()

    preds = model.generate(
      inp_ids,
      decoder_start_token_id=0,
      min_length=args.min_target_length,
      max_length=args.max_target_length*2,
      num_beams=args.num_beams,
      no_repeat_ngram_size=args.no_repeat_ngram_size,
      repetition_penalty=args.repetition_penalty,
      length_penalty=args.length_penalty,
    )

    for pred in preds:
      f.write(tokenizer.decode(pred) + '\n')
  f.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('-mode', default='train', type=str)

  parser.add_argument('-dataset', default='amazon', type=str)
  parser.add_argument('-num_aspects', default=18, type=int)
  parser.add_argument('-model_name', default='naive', type=str)
  parser.add_argument('-load_model', default=None, type=str)
  parser.add_argument('-seed_type', default='my', type=str)

  parser.add_argument('-train_file', default=None, type=str)
  parser.add_argument('-asp_dev_file', default=None, type=str)
  parser.add_argument('-gen_dev_file', default=None, type=str)
  parser.add_argument('-asp_test_file', default=None, type=str)
  parser.add_argument('-gen_test_file', default=None, type=str)

  parser.add_argument('-data_dir', default='data', type=str)
  parser.add_argument('-model_dir', default='model', type=str)

  parser.add_argument('-model_type', default='t5-small', type=str)
  parser.add_argument('-model_dim', default=512, type=int)
  parser.add_argument('-use_keywords', default='input', type=str) # none, input, output
  parser.add_argument('-use_switch', default='input', type=str) # none, input, output

  parser.add_argument('-batch_size', default=16, type=int)
  parser.add_argument('-learning_rate', default=1e-6, type=float)
  parser.add_argument('-no_train_steps', default=500000, type=int)
  parser.add_argument('-no_warmup_steps', default=20000, type=int)
  parser.add_argument('-check_every', default=1000, type=int)
  parser.add_argument('-ckpt_every', default=10000, type=int)

  parser.add_argument('-max_length', default=512, type=int)

  parser.add_argument('-min_target_length', default=15, type=int)
  parser.add_argument('-max_target_length', default=128, type=int)
  parser.add_argument('-num_beams', default=2, type=int)
  parser.add_argument('-no_repeat_ngram_size', default=3, type=int)
  parser.add_argument('-repetition_penalty', default=1, type=float)
  parser.add_argument('-length_penalty', default=1, type=float)

  args = parser.parse_args()
  if args.mode == 'train':
    train(args)
  elif args.mode == 'eval-general':
    evaluate(args, 'general')
  elif args.mode == 'eval-aspect':
    evaluate(args, 'aspect')
  elif args.mode == 'eval-multi':
    evaluate(args, 'multi')
  elif args.mode == 'eval-double':
    evaluate(args, 'double')