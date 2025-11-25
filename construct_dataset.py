import torch
import numpy as np

VALID = set(["A", "C", "G", "T"])
N = 61936*2
inv = 0
pos_count = 0
neg_count = 0
neg_inv_lengths = []
below_50 = 0

sequences = []
labels = []

# Positive samples
for i in range(61936):
    path = f"human_enhancers_ensembl/train/positive/{i}.txt"
    with open(path, "r") as f:
        seq = f.readline().strip()

    # Minimum sequence length of 50
    skip = False
    if len(seq) < 50: 
        skip = True
        below_50 += 1

    if skip:
        continue

    # Validate characters
    for c in seq:
        if c not in VALID:
            raise ValueError(f"Invalid character '{c}' in {path}")

    sequences.append(seq)
    labels.append(1)
    pos_count += 1

# Negative samples
for i in range(61936):
    path = f"human_enhancers_ensembl/train/negative/{i}.txt"
    with open(path, "r") as f:
        seq = f.readline().strip()

    skip = False
    # use only sequences of length >= 50
    if len(seq) < 50: 
        skip = True
        below_50 += 1
    
    if skip:
        continue

    skip = False
    for c in seq:
        if c not in VALID:
            #print(f"Invalid character '{c}' in {path}")
            inv += 1
            skip = True
            neg_inv_lengths.append(len([c for c in seq if c == 'N']))
            break

    if skip:
        continue
        
    sequences.append(seq)
    labels.append(0)
    neg_count += 1

torch.save({"seqs": sequences, "labels": labels}, "train_raw.pt")
print('Saved dataset to train_raw.pt')
print(f'{inv} files found with invalid characters.')
print(f'{pos_count} positive sequences saved.')
print(f'{neg_count} negative sequences saved.')
print(f'Mean count of N characters in invalid files: {np.mean(np.array(neg_inv_lengths)):2f}')
print(f'{below_50} sequences with length < 50 removed from dataset.')
