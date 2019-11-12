from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np

alp_path = '/home/shut/try/logo_alp/adidas176.png'
save_path = alp_path[:-4]+'_mask.png'
alp_logo = Image.open(alp_path).convert('RGBA')
r, g, b, alp = alp_logo.split()
alp_t = transforms.ToTensor()(alp)
alp_t = alp_t.squeeze()
mask = alp_t * 255
mask[mask>10] = 255
mask[mask<=10] = 0
plt.imshow(mask.numpy(), cmap='gray')
