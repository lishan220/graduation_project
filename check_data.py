import json
import random

with open('data/processed/training/train_seq2seq.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for i in range(5):
    s = random.choice(data)
    print('input:', s['input'][:100])
    print('output:', s['output'][:200])
    print('---')