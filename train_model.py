import time
import os

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim

from EnhancerDataset import EnhancerDataset, enhancer_collate_fn
from EnhancerAttention import EnhancerAttention

import numpy as np
import torch

# for memory usage on CPU
try:
    import psutil
    USE_PSUTIL = True
except ImportError:
    USE_PSUTIL = False

# hopefully can use gpa soon lol
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

full_train_dataset = EnhancerDataset(data_type="train")

train_loader = DataLoader(
    full_train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=enhancer_collate_fn,
)


embed_size = 100  # dna2vec dim
model = EnhancerAttention(embed_size=embed_size).to(device)

loss_fn = nn.BCEWithLogitsLoss()
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop (benchmark run)
epochs = 5 # for preliminary benchmarking
print_every = 100 # steps

start_time = time.time()
if USE_PSUTIL:
    process = psutil.Process(os.getpid())

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for step, (inputs, masks, labels) in enumerate(train_loader, start=1):
        inputs = inputs.to(device) # (batch, N, embed_dim)
        masks = masks.to(device) if masks is not None else None
        labels = labels.to(device).float() # (batch,)

        # Forward
        logits = model(inputs, masks) # (batch,)
        loss = loss_fn(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if step % print_every == 0:
            avg_loss = running_loss / print_every
            if USE_PSUTIL:
                mem_mb = process.memory_info().rss / (1024 ** 2)
                print(
                    f"Epoch [{epoch+1}/{epochs}] Step [{step}] "
                    f"- Loss: {avg_loss:.4f} - RAM: {mem_mb:.1f} MB"
                )
            else:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{step}] - Loss: {avg_loss:.4f}")
            running_loss = 0.0

torch.save(model, 'preliminary_model.pth')

total_time = time.time() - start_time
print(f"Benchmark training complete in {total_time:.2f} seconds.")
