from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np

mask_path = '/home/shut/try/logo_alp/adidas176_mask.png'
mask = Image.open(mask_path).convert('L')
plt.subplot(221)
plt.imshow(mask, cmap='gray')

mask_t = transforms.ToTensor()(mask)
mask_t = mask_t.unsqueeze(0)  # torch.Size([1, 1, 176, 176])
conv1 = nn.Conv2d(1, 1, 3, 1, 1)
a = torch.ones((1, 1, 3, 3), dtype=torch.float)
a[0, 0, 1, 1] = 0
conv1.weight.data = a
maxpool1 = nn.MaxPool2d(3, 1, 1)
m1 = conv1(mask_t)  # torch.Size([1, 1, 176, 176])
contour_t = maxpool1(m1)  # torch.Size([1, 1, 176, 176])
contour_np = contour_t.squeeze().data.numpy()  # <class 'tuple'>: (176, 176)
plt.subplot(222)
plt.imshow(contour_np, cmap='gray')

mask_np = mask_t.squeeze().data.numpy()
contour = (1 - mask_np) * contour_np
plt.subplot(223)
plt.imshow(1 - mask_np, cmap='gray')
plt.subplot(224)
plt.imshow(contour, cmap='gray')
# plt.imsave(save_path, mask.numpy(), cmap='gray')
plt.show()
