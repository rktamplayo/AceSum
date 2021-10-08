import json
import numpy as np
import os
from tqdm import tqdm
import random

def create_data(filedir):
  print(filedir)

  # get aspects and keywords
  files = os.listdir(filedir + '/seeds/')
  keywords_dict = {}
  for file in files:
    f = open(filedir + '/seeds/' + file, 'r')
    keywords = []
    for _ in range(5):
      keyword = f.readline().strip().split()[-1]
      keywords.append(keyword)

    f.close()
    aspect = file.replace('.txt', '')
    keywords_dict[aspect] = keywords

  aspects = list(keywords_dict.keys()) # + ['general']

  instance_dict = {}
  f = open(filedir + '/train.jsonl', 'r')
  for line in tqdm(f):
    inst = json.loads(line.strip())

    domain = 'space'
    if domain not in instance_dict:
      instance_dict[domain] = {}

    reviews = inst['reviews']
    for review in reviews :
      sentences = review['sentences']

      # sanity check
      if len(sentences) > 35 or len(sentences) < 1:
        continue
      if max([len(sentence.split()) for sentence in sentences]) > 35:
        continue

      review = ' '.join(sentences).split()
      
      # check whether aspect keywords in review
      class_list = []
      for aspect in aspects:
        keywords = keywords_dict[aspect]
        includes = int(any([keyword in review for keyword in keywords]))
        class_list.append(includes)
      #assert len(class_list) == 3

      # add general class
      # if any(class_list):
      #   class_list.append(0)
      # else:
      #   class_list.append(1)

      # add review to corresponding aspect buckets
      instance_tuple = (sentences, class_list)
      class_list = tuple(class_list)
      if class_list not in instance_dict[domain]:
        instance_dict[domain][class_list] = []
      instance_dict[domain][class_list].append(instance_tuple)

  f.close()

  for domain in instance_dict:
    print('domain', domain)

    lengths = [len(instance_dict[domain][key]) for key in instance_dict[domain]]
    print(lengths)
    min_length = sorted(lengths)[1]

    for i in range(len(aspects)):
      c = [0] * len(aspects)
      c[i] = 1
      print(c)
      print(len(instance_dict[domain][tuple(c)]))

    print('mininum instances per tuple', min_length)

    data = []
    for key in instance_dict[domain]:
      instances = instance_dict[domain][key]
      random.shuffle(instances)
      data += instances[:min_length]

    print('total data', len(data))
    random.shuffle(data)
    max_text_length = 0

    domain_aspects = aspects

    f = open(filedir + '/train.mil.jsonl', 'w')
    count_dict = {aspect:0 for aspect in domain_aspects}
    for inst in data:
      new_inst = {}
      new_inst['review'] = inst[0]
      max_text_length = max(max_text_length, len(inst[0]))
      class_dict = {}

      for i, aspect in enumerate(domain_aspects):
        class_dict[aspect] = 'yes' if inst[1][i] else 'no'
        if inst[1][i]:
          count_dict[aspect] += 1
      new_inst['aspects'] = class_dict
      f.write(json.dumps(new_inst) + '\n')

    f.close()

    print('max text length', max_text_length)
    print(count_dict)

create_data('data/oposum/bag')
create_data('data/oposum/boots')
create_data('data/oposum/bt')
create_data('data/oposum/keyboard')
create_data('data/oposum/tv')
create_data('data/oposum/vacuum')
#create_data('data/space/')