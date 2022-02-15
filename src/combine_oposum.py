import json
import sys
from tqdm import tqdm

file = sys.argv[1]

domains = ['bag', 'boots', 'bt', 'keyboard', 'tv', 'vacuum']

f2 = open(file, 'w')
for i, domain in enumerate(domains):
   f = open('data/oposum/' + domain + '/' + file, 'r')
   data = [json.loads(line.strip()) for line in f]
   f.close()

   for inst in tqdm(data):
      switch = inst['switch']
      new_switch = []
      for j in range(len(domains)):
         if i == j:
            new_switch += switch
         else:
            new_switch += [-1] * 3
      inst['switch'] = new_switch
      f2.write(json.dumps(inst) + '\n')


f2.close()
