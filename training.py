import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import os
from torchvision import transforms


# =========================
# SETTINGS
# =========================
train = True          # Set to False to generate images only
image_paths = [
    "image.png","image1.png","image2.png","image3.png","image4.png",
    "image5.png","image6.png","image7.png","image8.png","image9.png",
    "image10.png","image11.png","image12.png","image13.png","image14.png",
    "image15.png","image16.png","image17.png","image18.png","image19.png",
    "image20.png","image21.png","image22.png", "image23.png"
]
device = "cuda" if torch.cuda.is_available() else "cpu"
T = 400
epochs = 900
batch_size = 4
lr = 2e-4
ema_decay = 0.993


augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
    ])
# =========================
# LOAD IMAGES
# =========================
def load_images(paths, size=64):
    images = []
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: {path} not found")
            continue
        img = Image.open(path).convert("RGB").resize((size, size))
        x = np.array(img).astype(np.float32) / 127.5 - 1
        x = np.transpose(x, (2, 0, 1))
        images.append(x)
    images = np.stack(images)
    return torch.tensor(images)

dataset = load_images(image_paths, size=64)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# =========================
# COSINE BETA SCHEDULE
# =========================
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

betas = cosine_beta_schedule(T).to(device)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

# =========================
# MODEL DEFINITION (UNet + helpers)
# =========================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch*2)
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, t):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        scale_shift = self.time_mlp(t)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(h)
        h = self.conv2(h)
        return self.residual(x) + h

class MultiHeadAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels*3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(B, self.heads, C//self.heads, H*W)
        k = k.reshape(B, self.heads, C//self.heads, H*W)
        v = v.reshape(B, self.heads, C//self.heads, H*W)
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)

class UNet(nn.Module):
    def __init__(self, base=128, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim*4),
            nn.SiLU(),
            nn.Linear(time_dim*4, time_dim)
        )
        self.init = nn.Conv2d(3, base, 3, padding=1)
        self.down1_block1 = ResBlock(base, base, time_dim)
        self.down1_block2 = ResBlock(base, base, time_dim)
        self.downsample1 = nn.Conv2d(base, base*2, 4, 2, 1)
        self.down2_block1 = ResBlock(base*2, base*2, time_dim)
        self.down2_attn = MultiHeadAttention(base*2)
        self.down2_block2 = ResBlock(base*2, base*2, time_dim)
        self.downsample2 = nn.Conv2d(base*2, base*4, 4, 2, 1)
        self.down3_block1 = ResBlock(base*4, base*4, time_dim)
        self.down3_attn = MultiHeadAttention(base*4)
        self.down3_block2 = ResBlock(base*4, base*4, time_dim)
        self.downsample3 = nn.Conv2d(base*4, base*8, 4, 2, 1)
        self.mid_block1 = ResBlock(base*8, base*8, time_dim)
        self.mid_attn = MultiHeadAttention(base*8)
        self.mid_block2 = ResBlock(base*8, base*8, time_dim)
        self.up1 = nn.ConvTranspose2d(base*8, base*4, 4, 2, 1)
        self.up1_block1 = ResBlock(base*8, base*4, time_dim)
        self.up1_block2 = ResBlock(base*4, base*4, time_dim)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 4, 2, 1)
        self.up2_block1 = ResBlock(base*4, base*2, time_dim)
        self.up2_block2 = ResBlock(base*2, base*2, time_dim)
        self.up3 = nn.ConvTranspose2d(base*2, base, 4, 2, 1)
        self.up3_block1 = ResBlock(base*2, base, time_dim)
        self.up3_block2 = ResBlock(base, base, time_dim)
        self.final = nn.Conv2d(base, 3, 1)
    def forward(self, x, t):
        t = self.time_mlp(t)
        x = self.init(x)
        d1 = self.down1_block1(x, t)
        d1 = self.down1_block2(d1, t)
        x = self.downsample1(d1)
        d2 = self.down2_block1(x, t)
        d2 = self.down2_attn(d2)
        d2 = self.down2_block2(d2, t)
        x = self.downsample2(d2)
        d3 = self.down3_block1(x, t)
        d3 = self.down3_attn(d3)
        d3 = self.down3_block2(d3, t)
        x = self.downsample3(d3)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        x = self.up1(x)
        x = torch.cat([x, d3], dim=1)
        x = self.up1_block1(x, t)
        x = self.up1_block2(x, t)
        x = self.up2(x)
        x = torch.cat([x, d2], dim=1)
        x = self.up2_block1(x, t)
        x = self.up2_block2(x, t)
        x = self.up3(x)
        x = torch.cat([x, d1], dim=1)
        x = self.up3_block1(x, t)
        x = self.up3_block2(x, t)
        return self.final(x)

# =========================
# HELPER FUNCTIONS
# =========================
def update_ema(ema_model, model):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data = ema_decay * ema_p.data + (1 - ema_decay) * p.data


@torch.no_grad()
def sample(model, shape):
    x = torch.randn(shape, device=device)
    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, device=device).float()
        t_norm = t_batch / T  # ← must match training normalization
        pred_noise = model(x, t_norm)

        alpha = alphas[t]
        alpha_bar = alpha_bars[t]
        alpha_bar_prev = alpha_bars[t - 1] if t > 0 else torch.tensor(1.0)

        # Corrected DDPM denoising step
        x0_pred = (x - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
        x0_pred = x0_pred.clamp(-1, 1)

        # Posterior mean
        mean = (torch.sqrt(alpha_bar_prev) * (1 - alpha) * x0_pred +
                torch.sqrt(alpha) * (1 - alpha_bar_prev) * x) / (1 - alpha_bar)

        # Fix #2: add noise for t > 0
        if t > 0:
            variance = (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha)
            x = mean + torch.sqrt(variance) * torch.randn_like(x)
        else:
            x = mean
    return x

# =========================
# MODEL SETUP
# =========================
model = UNet().to(device)
ema_model = UNet().to(device)
ema_model.load_state_dict(model.state_dict())
optimizer = optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# =========================
# TRAINING OR GENERATION
# =========================
if train:
    for epoch in range(epochs):
        for x0 in dataloader:
            x0 = augment(x0.to(device))
            B = x0.size(0)
            t = torch.randint(0, T, (B,), device=device)
            noise = torch.randn_like(x0)
            alpha_bar = alpha_bars[t].view(B,1,1,1)
            xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
            t_norm = t / T
            pred = model(xt, t_norm.float())
            loss = loss_fn(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(ema_model, model)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss {loss.item():.4f}")
    torch.save(ema_model.state_dict(), "diffusion_ema.pth")
    print("Training done. Model saved.")

else:
    # Load pretrained EMA model
    model.load_state_dict(torch.load("diffusion_ema.pth", map_location=device))
    model.eval()
    num_images = 6
    samples = sample(model, (num_images, 3, 64, 64))
    samples = samples.clamp(-1,1)
    samples = (samples.cpu().numpy().transpose(0,2,3,1) + 1)/2
    plt.figure(figsize=(12,6))
    for i in range(num_images):
        plt.subplot(2,3,i+1)
        plt.imshow(samples[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()