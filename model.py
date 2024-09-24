""" Parts of the U-Net model """
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import cv2 as cv
import copy

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return self.sigmoid(logits)
    
def load_unet(filename:str, grayscale:bool):
    unet = UNet(1 if grayscale else 3, 1)
    with open(filename, 'rb') as f:
        unet.load_state_dict(torch.load(f))
    if torch.backends.mps.is_available():
        unet.to('mps')
    return unet

def predict_mask(unet:UNet, img, threshold = 0.5):
    # grayscale
    if unet.n_channels == 1:
        if len(img.shape) > 2 and img.shape[2] > 1:
            # img is not grayscale, but grayscale required -> BGR2GRAY
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        X = torch.tensor(img / 255., dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    else:
        X = torch.tensor(img / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    if torch.backends.mps.is_available():
        X = X.to('mps')

    unet.eval()
    with torch.no_grad():
        Y = unet(X)
    img_Y = Y.squeeze().cpu().numpy()

    img_Y = np.float32(img_Y >= threshold)

    contours, _ = cv.findContours(np.uint8(img_Y), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    points = __contours_to_point_list__(contours)

    polys = []
    min_area = 10.#300.
    img_test = np.zeros(img_Y.shape, dtype=np.float32)
    for poly in points:
        area = __poly_area__(poly)
        if area < min_area:
            continue
        polys.append(poly)
        img_test = __poly__(img_test, poly, color=(1,0,0))
        #cv.imshow('test', img_test)
    #cv.waitKey()

    points = __merge_ploys__(polys)
    #img_Y = cv.cvtColor(img_Y, cv.COLOR_GRAY2BGR)

    #img_Y = __poly__(img_Y, points, color=(255, 0, 0))

    # for p in points:
    #     img_Y = cv.circle(img_Y, (p[0], p[0]), radius=1, color=(255, 0, 0), thickness=-1)

    #cv.imshow('img_Y', img_Y)
    #cv.waitKey()
    points = __remove_single_points__(points)
    return points

def __contours_to_point_list__(contours):
    c_list = []
    for cont in contours:
        if len(cont) < 3:
            continue
        p_list = []
        for c in cont:
            p_list.append([c[0, 0], c[0, 1]])
        c_list.append(p_list)
    return c_list

def __poly_area__(points):
    p = np.array(points, dtype=np.int16)
    x = p[:, 0]
    y = p[:, 1]
    area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    return area


def __merge_ploys__(polys):
    poly = []
    previous_point = None
    min_dist = 10.
    for i, cont in enumerate(polys):
        poly.append([0, 0])
        for p in cont:
            if previous_point is None:
                dist = min_dist
            else:
                dist = np.sqrt((p[0] - previous_point[0])**2 + (p[1] - previous_point[1])**2)
            if dist >= min_dist:
                poly.append(p)
                previous_point = copy.copy(p)
        p0 = copy.copy(cont[0])
        poly.append(p0)
        poly.append([0, 0])
        
        #if last_p_previous_cont is not None:
        #    poly.append(last_p_previous_cont)
        #last_p_previous_cont = copy.copy(p)
        
        #poly.append([0, 0])
    return poly

def __poly__(img, points, color):
    overlay = img.copy() 

    points = np.array(points)#.reshape((len(points), 2))
    overlay = cv.fillPoly(overlay, [points], color) 
    
    alpha = 0.4
    img = cv.addWeighted(overlay, alpha, img, 1 - alpha, 0) 

    return img

def __remove_single_points__(poly):
    points = []
    for i, p in enumerate(poly):
        if i > 0 and i < len(poly) - 1 and poly[i-1][0] == poly[i+1][0] and poly[i-1][1] == poly[i+1][1]:
            continue
        points.append([int(p[0]), int(p[1])])
    return points