#!/usr/bin/env python3
import numpy as np
from pathlib import Path

emb_dir = Path('data/processed')
corrupted_list = []
total_count = 0

print('Scanning embeddings...')
for f in sorted(emb_dir.glob('*_emb.npy')):
    total_count += 1
    try:
        arr = np.load(f)
    except Exception as e:
        corrupted_list.append(f.name)

print(f'Total: {total_count}')
print(f'Corrupted: {len(corrupted_list)}')

if corrupted_list:
    print('\nCorrupted files:')
    for name in corrupted_list[:50]:
        print(f'  {name}')
