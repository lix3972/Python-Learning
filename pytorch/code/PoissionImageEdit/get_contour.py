from PIL import Image

import torchvision.transforms as transforms

import torch

import torch.nn as nn

from matplotlib import pyplot as plt

import numpy as np





class MyCap(torch.autograd.Function):

    """

    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html#sphx-glr-beginner-examples-autograd-two-layer-net-custom-function-py

    在ReLU的基础上增加 > 1 输出 1

    """

    @staticmethod

    def forward(ctx, input):

        ctx.save_for_backward(input)

        return input.clamp(min=0, max=1)



    @staticmethod

    def backward(ctx, grad_output):

        input, = ctx.saved_tensors

        grad_input = grad_output.clone()

        grad_input[input < 0] = 0

        grad_input[input > 1] = 0

        return grad_input





# mask_path = '/home/shut/try/logo_alp/adidas176_mask.png'

# mask = Image.open(mask_path).convert('L')

# plt.imshow(mask, cmap='gray')

# mask_t = transforms.ToTensor()(mask)

bg = torch.zeros((10, 10), dtype=torch.float)

fg = torch.ones(4, 4)

com = bg

com[3:7, 3:7] = fg

mask = com

plt.subplot(221)

plt.imshow(mask.data.numpy(), cmap='gray')



mask_t = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 10, 10] ; torch.Size([1, 1, 176, 176])

conv1 = nn.Conv2d(1, 1, 3, 1, 1)

a = torch.ones((1, 1, 3, 3), dtype=torch.float)

a[0, 0, 1, 1] = 0

# from Pytorch STN instance

# Initialize the weights/bias with identity transformation

#         self.fc_loc[2].weight.data.zero_()

#         self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

conv1.weight.data = a  # or: conv1.weight.data.copy_(a)

conv1.bias.data.zero_()

# maxpool1 = nn.MaxPool2d(3, 1, 1)

my_cap = MyCap.apply


# 开始与tmp不同
mask_inv = 1-mask_t

m1 = conv1(mask_inv)  # torch.Size([1, 1, 176, 176])

plt.subplot(223)

plt.imshow(m1.squeeze().data.numpy(), cmap='gray')



# contour_t = maxpool1(m1)  # torch.Size([1, 1, 176, 176])

contour_t = my_cap(m1)

contour_np = contour_t.squeeze().data.numpy()  # <class 'tuple'>: (176, 176)

plt.subplot(222)

plt.imshow(contour_np, cmap='gray')



mask_np = mask_t.squeeze().data.numpy()

contour = mask_np * contour_np

# plt.subplot(223)

# plt.imshow(1 - mask_np, cmap='gray')

plt.subplot(224)

plt.imshow(contour, cmap='gray')

# plt.imsave(save_path, mask.numpy(), cmap='gray')

plt.show()
