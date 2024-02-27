#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 15:11:54 2024

@author: Atharva Tyagi
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
# from datasets import VanillaUnetDataset
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(out_channels, out_channels, 3, 1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout2d(0.5),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super(UNET, self).__init__()
        self.comnDiff = features[1] - features[0]
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features[:]):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature+self.comnDiff, feature+self.comnDiff, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv((feature+self.comnDiff)+feature, feature))
            
        self.bottleneck = DoubleConv(features[-1], features[-1]+self.comnDiff)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)



    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# def test():
#     dataset = VanillaUnetDataset("../tmpdat")
#     dataloader = DataLoader(dataset, batch_size=2, pin_memory=True)
#     for batch, target, dates in dataloader:
#         print(f"{dates}")
#         x = batch
#         break
#     x = x.to("cuda")
#     model = UNET(in_channels=287, out_channels=12, features=[288, 352, 416, 480]).to("cuda")
#     preds = model(x)
#     print(preds.shape)
#     inslice = preds[0][0][:][:]
#     inslice = inslice.cpu().detach()
#     plt.imshow(inslice.numpy(), cmap='viridis', origin='lower')
#     plt.colorbar()
#     plt.show()
#     print("all good")

if __name__ == "__main__":
    pass#test()