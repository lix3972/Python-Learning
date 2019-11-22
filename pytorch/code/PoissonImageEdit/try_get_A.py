from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np


class MyTruncate(torch.autograd.Function):
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
        return grad_input


class MyNonzero(torch.autograd.Function):
    """
    input的零值反向传播时梯度为0
    """

    # def __init__(self):
    #     super(MyNonzero, self).__init__()

    @staticmethod
    def forward(ctx, input):
        omega = torch.nonzero(input)
        ctx.save_for_backward(input, omega)
        return omega

    @staticmethod
    def backward(ctx, grad_output):
        input, omega = ctx.saved_tensors
        grad_tmp = grad_output.clone()
        grad_input = torch.zeros(input.shape)
        for n in range(omega.shape[0]):
            y, x = omega[n]
            grad_input[y, x] = torch.max(grad_tmp[n])
        return grad_input


def check_existence(mask, id_h, id_w):
    h, w = mask.shape
    if 0 <= id_h and id_h <= h - 1 and 0 <= id_w and id_w <= w - 1:
        if mask[id_h][id_w] == 1:
            return bool(1)

        else:
            return bool(0)

    else:
        return bool(0)


# 构造mask
bg = torch.zeros((10, 10), dtype=torch.float)
fg = torch.ones(4, 4)
com = bg
com[3:7, 3:7] = fg
mask = com
mask.requires_grad_(True)
# plt.subplot(221)
# plt.imshow(mask.data.numpy(), cmap='gray')
# =========== apply my class ==============
# get_A = GetA.apply
# # A = torch.sparse.FloatTensor(16, 16).to_dense()
# A, omega, omega_yx, ngb_flag = get_A(mask)
# A.backward(torch.ones(A.shape))
# print('get A pass')

my_nonzero = MyNonzero.apply

# ============ try ========================
h, w = mask.shape
# omega = torch.nonzero(mask)
omega = my_nonzero(mask)
# ====== 测试是否可以反向传播 =================
omega.backward(torch.ones(omega.shape, dtype=torch.long))
print('omega pass')

# y = torch.reshape(omega[:,0],[omega[:,0].shape[0],1])  # y = np.reshape(omega[0], [omega[0].shape[0], 1])
# x = torch.reshape(omega[:,1], [omega[:,1].shape[0], 1])  # x = np.reshape(omega[1], [omega[1].shape[0], 1])
# omega_list = torch.cat([y, x], 1)  # omega_list = np.concatenate([y, x], 1)

ngb_flag = []
omega_yx = torch.zeros((h, w), dtype=torch.int32)  # omega_yx = np.zeros((h, w), dtype=np.int32)
for index in range(omega.shape[0]):
    ## pixel location
    i, j = omega[index]

    ## create neigbourhoods flag
    ngb_flag.append([check_existence(mask, i, j + 1),
                     check_existence(mask, i, j - 1),
                     check_existence(mask, i + 1, j),
                     check_existence(mask, i - 1, j), ])

    ## store index to dictionary
    omega_yx[i][j] = index
ngb_flag = torch.tensor(ngb_flag, dtype=torch.uint8)

N = omega.shape[0]
A = torch.sparse.FloatTensor(N, N).to_dense()
for i in range(N):

    ## progress
    # progress_bar(i, N-1)

    ## fill 4 or -1
    ## center
    A[i, i] = 4
    id_h, id_w = omega[i]

    ## right
    if (ngb_flag[i][0]):
        j = omega_yx[id_h][id_w + 1]
        A[i, j] = -1

    ## left
    if (ngb_flag[i][1]):
        j = omega_yx[id_h][id_w - 1]
        A[i, j] = -1

    ## bottom
    if (ngb_flag[i][2]):
        j = omega_yx[id_h + 1][id_w]
        A[i, j] = -1

    ## up
    if (ngb_flag[i][3]):
        j = omega_yx[id_h - 1][id_w]
        A[i, j] = -1

A.backward(torch.ones(A.shape))

print('ok')

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
my_truncate = MyTruncate.apply
m1 = conv1(mask_t)  # torch.Size([1, 1, 176, 176])
plt.subplot(223)
plt.imshow(m1.squeeze().data.numpy(), cmap='gray')

# contour_t = maxpool1(m1)  # torch.Size([1, 1, 176, 176])
contour_t = my_truncate(m1)
contour_np = contour_t.squeeze().data.numpy()  # <class 'tuple'>: (176, 176)
plt.subplot(222)
plt.imshow(contour_np, cmap='gray')

mask_np = mask_t.squeeze().data.numpy()
contour = (1 - mask_np) * contour_np
# plt.subplot(223)
# plt.imshow(1 - mask_np, cmap='gray')
plt.subplot(224)
plt.imshow(contour, cmap='gray')
# plt.imsave(save_path, mask.numpy(), cmap='gray')
plt.show()

# =============================================================================
class GetA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = mask
        ctx.save_for_backward(input)
        h, w = mask.shape
        omega = torch.nonzero(mask)

        ngb_flag = []
        omega_yx = torch.zeros((h, w), dtype=torch.int32)  # omega_yx = np.zeros((h, w), dtype=np.int32)
        for index in range(omega.shape[0]):
            ## pixel location
            i, j = omega[index]

            ## create neigbourhoods flag
            ngb_flag.append([check_existence(mask, i, j + 1),
                             check_existence(mask, i, j - 1),
                             check_existence(mask, i + 1, j),
                             check_existence(mask, i - 1, j), ])

            ## store index to dictionary
            omega_yx[i][j] = index
        ngb_flag = torch.tensor(ngb_flag, dtype=torch.uint8)

        N = omega.shape[0]
        A = torch.sparse.FloatTensor(N, N).to_dense()
        for i in range(N):

            ## progress
            # progress_bar(i, N-1)

            ## fill 4 or -1
            ## center
            A[i, i] = 4
            id_h, id_w = omega[i]

            ## right
            if (ngb_flag[i][0]):
                j = omega_yx[id_h][id_w + 1]
                A[i, j] = -1

            ## left
            if (ngb_flag[i][1]):
                j = omega_yx[id_h][id_w - 1]
                A[i, j] = -1

            ## bottom
            if (ngb_flag[i][2]):
                j = omega_yx[id_h + 1][id_w]
                A[i, j] = -1

            ## up
            if (ngb_flag[i][3]):
                j = omega_yx[id_h - 1][id_w]
                A[i, j] = -1
        # return A, omega, omega_yx, ngb_flag
        # return A


    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = input
        return grad_input
