#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 15:11:54 2024

@author: Atharva Tyagi
"""

import torch, os
import torch.nn as nn
import torchvision.transforms.functional as TF
from coordconv import CoordConv2d
from datasets import TemporalUnetDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            CoordConv2d(in_channels, out_channels, 3, 1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout2d(0.5),
            CoordConv2d(out_channels, out_channels, 3, 1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout2d(0.5),
        )
        
    def forward(self, x):
        return self.conv(x)

class TemporalUnet(nn.Module):
    def __init__(self, in_channels, out_channels, features, timesteps):
        super(TemporalUnet, self).__init__()
        self.comnMult = features[1]/features[0]
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.timesteps = timesteps

        for _ in range(self.timesteps):
            down_t = nn.ModuleList()
            enchan = in_channels
            for feature in features:
                down_t.append(DoubleConv(enchan, feature))
                enchan = feature
            self.downs.append(down_t)
        
        otc = 0
        for i, feature in enumerate(reversed(features[:])):
            inc = feature*((self.timesteps+(self.comnMult*2))/(pow(2, i)))
            otc = inc/(self.comnMult*2)
            self.ups.append(
                nn.ConvTranspose2d(
                    int(inc), int(inc), kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(int(inc+(feature*self.timesteps)), int(otc)))
            
        self.bottleneck = DoubleConv(features[-1]*self.timesteps, int((features[-1]*self.comnMult)*self.timesteps))
        self.final_conv = nn.Conv2d(int(otc), int(out_channels), kernel_size=1)


    def forward(self, x):
        skip_connections = [[] for _ in range(len(self.downs[0]))]
        lagtimes = []
        
        for i in range(self.timesteps):
            w = x[:,i,:,:,:]
            for level, down in enumerate(self.downs[i]):
                w = down(w)
                skip_connections[level].append(w)
                w = self.pool(w)
            lagtimes.append(w)
        
        x = torch.cat(lagtimes, dim=1)
        x = self.bottleneck(x)
        skip_connections.reverse()

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connections_idx = skip_connections[idx//2][:]
            skip_connection = torch.cat(skip_connections_idx, dim=1)

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# def test():
#     dataset = TemporalUnetDataset("../tmpdat")
#     dataloader = DataLoader(dataset, batch_size=4, pin_memory=True)
#     for batch, target, dates in dataloader:
#         print(f"{dates}")
#         x = batch
#         break
#     x = x.to("cuda")
#     model = TemporalUnet(in_channels=28, out_channels=12, features=[32, 48, 72, 108], timesteps=6).to("cuda", non_blocking=True)
#     with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
#         preds = model(x)
#     print(preds.shape)
#     # inslice = preds[0][0][:][:]
#     # inslice = inslice.cpu().detach()
#     # plt.imshow(inslice.numpy(), cmap='viridis', origin='lower')
#     # plt.colorbar()
#     # plt.show()
#     print("all good")

# if __name__ == "__main__":
#     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
#     test()
