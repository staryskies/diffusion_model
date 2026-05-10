import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import os
from torchvision import transforms
from collections import OrderedDict

from transformers import CLIPTextModel, CLIPTokenizer

image_caption_dict = {
    "image.png": "Blue haired girl",
    "image1.png": "Blue haired girl",
    "image2.png": "Blue haired girl",
    "image3.png": "Blue haired girl",
    "image4.png": "Blue haired girl",
    "image5.png": "Blue haired girl",
    "image6.png": "Blue haired girl",
    "image7.png": "Blue haired girl",
    "image8.png": "Blue haired girl",
    "image9.png": "Blue haired girl",
    "image10.png": "Blue haired girl",
    "image11.png": "Blue haired girl",
    "image12.png": "Blue haired girl",
    "image13.png": "Blue haired girl",
    "image14.png": "Blue haired girl",
    "image15.png": "Blue haired girl",
    "image16.png": "Blue haired girl",
    "image17.png": "Blue haired girl",
    "image18.png": "Blue haired girl",
    "image19.png": "Blue haired girl",
    "image20.png": "Blue haired girl",
    "image21.png": "Blue haired girl",
    "image22.png": "Blue haired girl",
    "image copy.png": "Brown haired boy",
    "image copy 2.png": "Brown haired boy",
    "image copy 3.png": "Brown haired boy",
    "image copy 4.png": "Brown haired boy",
    "image copy 5.png": "Brown haired boy",
    "image copy 6.png": "Brown haired boy",
    "image copy 7.png": "Brown haired boy",
    "image copy 8.png": "Brown haired boy",
    "image copy 9.png": "Brown haired boy",
    "image copy 10.png": "Brown haired boy",
    "image copy 11.png": "Brown haired boy",
    "image copy 12.png": "Brown haired boy",
    "image copy 13.png": "Brown haired boy",
    "image copy 14.png": "Brown haired boy",
    "image copy 15.png": "Brown haired boy",
    "image copy 16.png": "Brown haired boy",
    "image copy 17.png": "Brown haired boy",
    "image copy 18.png": "Brown haired boy",
    "image copy 19.png": "Brown haired boy",
    "image copy 20.png": "Brown haired boy"
}


class ImageDatasetWcaption(image_caption_dict):
    def __init__(self, image_caption_dict):
        self.image_caption_dict = image_caption_dict
        self.cur_image = 0
        self.dict_keys = list(self.image_caption_dict.keys())
        self.size = 64
    def __len__(self):
        return len(self.image_caption_dict)
    def __next_image__(self):
        if self.len() < self.cur_image:
            return -1
        else:
            caption = self.image_caption_dict[self.dict_keys[self.cur_image]]
            img = Image.open(self.dict_keys[self.cur_image])
            img = img.resize((self.size, self.size))
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomRotation(15)(img)

            # H W C
            numPy_image = np.array(img).astype(np.float32) / 127.5 - 1

            torch_tensor = torch.tensor(torch.transpose(numPy_image, (2, 0, 1)))

            tokenizer = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt"
            )
            caption_tokens = tokenizer.torch_tensor.squeeze(0)

            return torch_tensor, caption_tokens
        
    





        
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

        #B = batch size
        #h = number of heads
        #d = channels per head = C // h
        #n = number of image tokens = H * W
        #k: (B, h, m, d) where
        #m = number of text tokens = 77
        #d = dimensions (same as head amount) 


        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v).reshape(B, C, H, W)
        return x + self.proj(out)
class CrossAttention(nn.Module):
    def __init__(self, channels, text_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (channels // heads) ** -0.5

        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_k = nn.Linear(text_dim, channels)
        self.to_v = nn.Linear(text_dim, channels)

        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x, text_emb):
        B, C, H, W = x.shape

        # IMAGE → queries (tokens = H*W)
        q = self.to_q(x).reshape(B, self.heads, C//self.heads, H*W)

        # TEXT → keys/values (tokens = 77)
        k = self.to_k(text_emb)  
        v = self.to_v(text_emb)

        k = k.reshape(B, -1, self.heads, C//self.heads).transpose(1,2)\
        v = v.reshape(B, -1, self.heads, C//self.heads).transpose(1,2)

   
        attn = torch.einsum('bhdn,bhmd->bhnm', q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum('bhnm,bhmd->bhdn', attn, v)

        out = out.reshape(B, C, H, W)
        return x + self.proj(out)