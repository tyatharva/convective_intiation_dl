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
import torch.nn as nn
import torch
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for input_batch, target_batch in dataloader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            output = model(input_batch)
            loss = criterion(output, target_batch)
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss


dataset = VanillaUnetDataset("../tmpdat")
dataloader = DataLoader(dataset, batch_size=2, pin_memory=True)

val_dataset = VanillaUnetDataset("../full_data/validate")
val_dataloader = DataLoader(val_dataset, batch_size=2, pin_memory=True)

model = UNET(in_channels=287, out_channels=12, features=[288, 352, 416, 480]).to(device)

pos_weight = torch.tensor([5.0]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (input_batch, target_batch, dates) in enumerate(dataloader):
        print(f"{dates}\n")
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)

        optimizer.zero_grad()
        output = model(input_batch)
        target_batch = target_batch.to(torch.float)
        loss = criterion(output, target_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(running_loss)
    
    # Update the learning rate based on the validation loss
    val_loss = validate(model, criterion, val_dataloader, device)
    scheduler.step(val_loss)

    print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(dataloader)} | Validation Loss: {val_loss:.6f}\n")

print("Training finished.")