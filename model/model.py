#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 21:52:53 2024

@author: Atharva Tyagi
"""

from torch.utils.data import DataLoader
from datasets import VanillaUnetDataset
from vanilla_unet import UNET
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.optim as optim
import time
st = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for input_batch, target_batch, dates in dataloader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            output = model(input_batch)
            target_batch = target_batch.to(torch.float)
            loss = criterion(output, target_batch)
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss


dataset = VanillaUnetDataset("../tmpdat")
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True)

val_dataset = VanillaUnetDataset("../full_data/validate")
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True, pin_memory=True)

model = UNET(in_channels=287, out_channels=12, features=[288, 352, 416, 480]).to(device)

pos_weight = torch.tensor([10000.0]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (input_batch, target_batch, dates) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)

        optimizer.zero_grad()
        output = model(input_batch)
        target_batch = target_batch.to(torch.float)
        loss = criterion(output, target_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f"\n\n\rDates: {dates}, Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}", end='')
    
    # Update the learning rate based on the validation loss
    val_loss = validate(model, criterion, val_dataloader, device)
    with open("loss.txt", "a") as file: file.write(f"Loss: {val_loss}\n")
    scheduler.step(val_loss)

    print(f"\nEpoch {epoch + 1}, Training Loss: {running_loss / len(dataloader)} | Validation Loss: {val_loss:.6f}\n")


torch.save(model, "model_full.pt")
torch.save(model.state_dict(), "model_dict.pth")
et = time.time()
tt = et - st
print(f"Training finished in {tt} seconds.")
