import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

T = 200  # increased so diffusion is visible
betas = np.linspace(0.0001, 0.02, T)

# Load image
image_path = 'image.png'
img_data = Image.open(image_path).convert("RGB")

# Normalize to [-1, 1]
x0 = np.array(img_data).astype(np.float32) / 127.5 - 1

x = x0.copy()
images = [x0.copy()]

# Forward diffusion
for t in range(T):
    noise = np.random.randn(*x.shape)
    x = np.sqrt(1 - betas[t]) * x + np.sqrt(betas[t]) * noise
    images.append(x.copy())

# Show gradual change
num_show = 10
indices = np.linspace(0, T, num_show, dtype=int)

fig, axes = plt.subplots(1, num_show, figsize=(20, 4))

for i, idx in enumerate(indices):
    img_display = np.clip((images[idx] + 1) / 2, 0, 1)
    axes[i].imshow(img_display)
    axes[i].set_title(f"t={idx}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()