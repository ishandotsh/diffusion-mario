import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import math 

# Dataset class to load our generated frames
class MarioFrameDataset(Dataset):
    def __init__(self, dataset_dir):
        self.current_frames_dir = os.path.join(dataset_dir, "current_frames")
        self.next_frames_dir = os.path.join(dataset_dir, "next_frames")
        self.actions = np.load(os.path.join(dataset_dir, "actions.npy"))
        self.frame_files = sorted(os.listdir(self.current_frames_dir))
        
    def __len__(self):
        return len(self.frame_files)
    
    def __getitem__(self, idx):
        # Load frames
        current_frame = cv2.imread(os.path.join(self.current_frames_dir, self.frame_files[idx]))
        next_frame = cv2.imread(os.path.join(self.next_frames_dir, self.frame_files[idx]))
        
        # Convert BGR to RGB and normalize to [-1, 1]
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1
        
        # Convert to torch tensors with correct memory format
        current_frame = torch.from_numpy(current_frame).permute(2, 0, 1).contiguous()
        next_frame = torch.from_numpy(next_frame).permute(2, 0, 1).contiguous()
        
        # Get action and convert to one-hot
        action = torch.zeros(7, dtype=torch.float32)  # SIMPLE_MOVEMENT has 7 actions
        action[self.actions[idx]] = 1
        
        return current_frame, action, next_frame

# U-Net blocks for the diffusion model
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )
        
        if self.residual:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        if self.residual:
            residual = self.residual_conv(x)
            return F.gelu(residual + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, residual=True)
        )
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels + out_channels, out_channels, residual=True)
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

# Main diffusion model
class MarioDiffusion(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_actions=7):
        super().__init__()
        self.time_dim = time_dim
        
        # Initial conv
        self.inc = DoubleConv(c_in, 64)
        
        # Action and time embeddings
        self.action_embedding = nn.Linear(num_actions, time_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Downsampling path
        self.down1 = Down(64, 128)     # 128x128 -> 64x64
        self.down2 = Down(128, 256)    # 64x64 -> 32x32
        self.down3 = Down(256, 512)    # 32x32 -> 16x16
        self.down4 = Down(512, 512)    # 16x16 -> 8x8
        
        # Bottleneck
        self.bot1 = DoubleConv(512, 512, residual=True)
        self.bot2 = DoubleConv(512, 512, residual=True)
        self.bot3 = DoubleConv(512, 512, residual=True)
        
        # Upsampling path
        self.up1 = Up(512, 512)    # 8x8 -> 16x16
        self.up2 = Up(512, 256)    # 16x16 -> 32x32
        self.up3 = Up(256, 128)    # 32x32 -> 64x64
        self.up4 = Up(128, 64)     # 64x64 -> 128x128
        
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def forward(self, x, t, action):
        # Embed time and action
        t = self.time_mlp(t)
        action_emb = self.action_embedding(action)
        t = t + action_emb
        
        # Downsampling
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)
        x5 = self.down4(x4, t)
        
        # Bottleneck
        x5 = self.bot1(x5)
        x5 = self.bot2(x5)
        x5 = self.bot3(x5)
        
        # Upsampling
        x = self.up1(x5, x4, t)
        x = self.up2(x, x3, t)
        x = self.up3(x, x2, t)
        x = self.up4(x, x1, t)
        
        return self.outc(x)

# Sinusoidal position embeddings for time
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings 