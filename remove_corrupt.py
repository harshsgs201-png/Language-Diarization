import numpy as np
from pathlib import Path

emb_dir = Path('data/processed')
corrupted_files = []
total = 0

print('=== SCANNING EMBEDDINGS ===\n')
for f in sorted(emb_dir.glob('*_emb.npy')):
    total += 1
    try:
        arr = np.load(f)
        if arr.size == 0 or arr.shape[0] == 0:
            corrupted_files.append((f, 'EMPTY_ARRAY'))
    except ValueError as e:
        err_msg = str(e)
        if 'could not read' in err_msg or 'Expected' in err_msg:
            corrupted_files.append((f, 'READ_ERROR'))
    except Exception as e:
        corrupted_files.append((f, 'OTHER_ERROR'))

print(f'Total embeddings scanned: {total}')
print(f'Corrupted embeddings found: {len(corrupted_files)}')

if corrupted_files:
    print('\n⚠️  CORRUPTED FILES (NOT DELETED YET):')
    for path, error_type in corrupted_files[:30]:
        print(f'  - {path.name} ({error_type})')
    
    print(f'\n🔴 STOP: {len(corrupted_files)} corrupted files detected!')
    print('These files need to be re-extracted.')
    print('Do NOT proceed with training until ALL are fixed.')
else:
    print('\n✅ No corrupted files found! All embeddings are valid.')
