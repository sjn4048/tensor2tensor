import os
import sys

assert len(sys.argv) == 2, 'A dataset name is needed.'
dataset = sys.argv[1]

train_en = f'data/{dataset}/train.en'
train_de = f'data/{dataset}/train.de'

texts_files = [train_de, train_en]

vocabs = set()
for file in texts_files:
  with open(file, 'r') as f:
    lines = f.readlines()
    for l in lines:
      for token in l.split(' '):
        vocabs.add(token.strip())

vocabs.add('UNK') # this is coincide with the settings in [xxxProblem] class.

print('vocab size: ', len(vocabs))
print('first of vocabs: ', list(vocabs)[:30])

output_dir = f'data/{dataset}'
filename = 'vocab.%d.subwords' % len(vocabs)
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

with open(os.path.join(output_dir, filename), 'w') as f:
  f.write('\n'.join(vocabs))
