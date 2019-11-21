from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from utils.image_show import plt_tensor, tensor2np_uint8


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


# =============================================================================
def lap_at_index(source, index, contour, ngb_flag):
    ## current location
    i, j = index
    ## take laplacian
    N = torch.sum(ngb_flag == True)
    val = (N * source[i, j]
           - (float(ngb_flag[0] == True) * source[i, j + 1])
           - (float(ngb_flag[1] == True) * source[i, j - 1])
           - (float(ngb_flag[2] == True) * source[i + 1, j])
           - (float(ngb_flag[3] == True) * source[i - 1, j]))
    return val


def constrain(target, index, contuor, ngb_flag):
    ## current location
    i, j = index
    ## In order to use "Dirichlet boundary condition",
    ## if on boundry, add in target intensity --> set constraint grad(source) = target at boundary
    if (contuor[i][j] == 1):
        val = ((ngb_flag[0] == 0).to(dtype=torch.float) * target[i, j + 1]
               + (ngb_flag[1] == 0).to(dtype=torch.float) * target[i, j - 1]
               + (ngb_flag[2] == 0).to(dtype=torch.float) * target[i + 1, j]
               + (ngb_flag[3] == 0).to(dtype=torch.float) * target[i - 1, j])
        return val
    ## If not on boundry, just take grad.
    else:
        val = torch.tensor(0.0)
        return val


def importing_gradients(src, tar, omega, contour, ngb_flag):
    ### output array
    u_r = torch.zeros(omega.shape[0])
    u_g = torch.zeros(omega.shape[0])
    u_b = torch.zeros(omega.shape[0])
    ### take laplacian
    for index in range(omega.shape[0]):
        ## apply each color channel
        u_r[index] = lap_at_index(src[:, :, 0], omega[index], contour, ngb_flag[index]) \
                      + constrain(tar[:, :, 0], omega[index], contour, ngb_flag[index])
        u_g[index] = lap_at_index(src[:, :, 1], omega[index], contour, ngb_flag[index]) \
                      + constrain(tar[:, :, 1], omega[index], contour, ngb_flag[index])
        u_b[index] = lap_at_index(src[:, :, 2], omega[index], contour, ngb_flag[index]) \
                      + constrain(tar[:, :, 2], omega[index], contour, ngb_flag[index])

    return u_r, u_g, u_b


def lap_at_index_mixing(source, target, index, contuor, ngb_flag):
    ## current location
    i, j = index

    ## gradient for source image
    grad_right_src = float(ngb_flag[0] == True) * (source[i, j] - source[i, j + 1])
    grad_left_src = float(ngb_flag[1] == True) * (source[i, j] - source[i, j - 1])
    grad_bottom_src = float(ngb_flag[2] == True) * (source[i, j] - source[i + 1, j])
    grad_up_src = float(ngb_flag[3] == True) * (source[i, j] - source[i - 1, j])

    ## gradient for target image
    grad_right_tar = float(ngb_flag[0] == True) * (target[i, j] - target[i, j + 1])
    grad_left_tar = float(ngb_flag[1] == True) * (target[i, j] - target[i, j - 1])
    grad_bottom_tar = float(ngb_flag[2] == True) * (target[i, j] - target[i + 1, j])
    grad_up_tar = float(ngb_flag[3] == True) * (target[i, j] - target[i - 1, j])

    val = [grad_right_src, grad_left_src, grad_bottom_src, grad_up_src]

    if (abs(grad_right_src) < abs(grad_right_tar)):
        val[0] = grad_right_tar

    if (abs(grad_left_src) < abs(grad_left_tar)):
        val[1] = grad_left_tar

    if (abs(grad_bottom_src) < abs(grad_bottom_tar)):
        val[2] = grad_bottom_tar

    if (abs(grad_up_src) < abs(grad_up_tar)):
        val[3] = grad_up_tar

    return val[0] + val[1] + val[2] + val[3]


def mixing_gradients(src, tar, omega, contour, ngb_flag):
  ### output array
  u_r = torch.zeros(omega.shape[0])
  u_g = torch.zeros(omega.shape[0])
  u_b = torch.zeros(omega.shape[0])


  ### take laplacian
  for index in range(omega.shape[0]):

    ## apply each color channel
    u_r[index] = lap_at_index_mixing(src[:, :, 0], tar[:, :, 0], omega[index], contour, ngb_flag[index]) \
                + constrain(tar[:, :, 0], omega[index], contour, ngb_flag[index])
    u_g[index] = lap_at_index_mixing(src[:, :, 1], tar[:, :, 1], omega[index], contour, ngb_flag[index]) \
                + constrain(tar[:, :, 1], omega[index], contour, ngb_flag[index])
    u_b[index] = lap_at_index_mixing(src[:, :, 2], tar[:, :, 2], omega[index], contour, ngb_flag[index]) \
                + constrain(tar[:, :, 2], omega[index], contour, ngb_flag[index])


  return u_r, u_g, u_b


class GetPIE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, mask, tar):  # mask size N, C, H, W
        ctx.save_for_backward(src, mask, tar)

        method = 'import'
        # method = 'mix'

        n, c, h, w = mask.shape
        if n > 1:
            print('The batch_size should be 1.')
            return
        # mask_t = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 10, 10] ; torch.Size([1, 1, 176, 176])
        conv1 = nn.Conv2d(1, 1, 3, 1, 1)
        a = torch.ones((1, 1, 3, 3), dtype=torch.float)
        a[0, 0, 1, 1] = 0
        # from Pytorch STN instance
        # Initialize the weights/bias with identity transformation
        #         self.fc_loc[2].weight.data.zero_()
        #         self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        conv1.weight.data = a  # or: conv1.weight.data.copy_(a)
        conv1.bias.data.zero_()
        my_cap = MyCap.apply
        m1 = conv1(mask)  # torch.Size([1, 1, 176, 176])
        erode_t = my_cap(m1)

        # ==== mask = mask.squeeze() ================
        mask = mask.squeeze()
        src = src.squeeze().permute(1, 2, 0)
        tar = tar.squeeze().permute(1, 2, 0)
        # contour = (1 - mask) * erode_t.squeeze()
        contour = mask * erode_t.squeeze()
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

        ## select process type
        if (method == "import"):
            u_r, u_g, u_b = importing_gradients(src, tar, omega, contour, ngb_flag)
        if (method == "mix"):
            u_b, u_g, u_r = mixing_gradients(src, tar, omega, contour, ngb_flag)

        Ainv = torch.inverse(A)
        x_r = u_r.unsqueeze(0).mm(Ainv)
        x_g = u_g.unsqueeze(0).mm(Ainv)
        x_b = u_b.unsqueeze(0).mm(Ainv)

        x_r, x_g, x_b = x_r.squeeze(0), x_g.squeeze(0), x_b.squeeze(0)

        blended = tar.clone()
        overlapped = tar.clone()

        for index in range(omega.shape[0]):
            i, j = omega[index]

            ## normal
            blended[i][j][0] = torch.clamp(x_b[index], 0.0, 1.0)
            blended[i][j][1] = torch.clamp(x_g[index], 0.0, 1.0)
            blended[i][j][2] = torch.clamp(x_r[index], 0.0, 1.0)

            ## overlapping
            overlapped[i][j][0] = src[i][j][0]
            overlapped[i][j][1] = src[i][j][1]
            overlapped[i][j][2] = src[i][j][2]

        return blended.permute(2, 0, 1)  # * 255, overlapped * 255

    @staticmethod
    def backward(ctx, grad_output):
        src, mask, tar = ctx.saved_tensors
        l1_grad = nn.L1Loss()(grad_output, torch.zeros(grad_output.shape))
        grad_stc = l1_grad * torch.ones(src.shape)
        grad_mask = l1_grad * torch.ones(mask.shape)
        grad_tar = l1_grad * torch.ones(tar.shape)
        return grad_stc, grad_mask, grad_tar


if __name__ == '__main__':
    method = 'mix'  # import, mix

    mask_path = '/home/lix/mySTN11/datasets/try/mask.png'
    mask0 = Image.open(mask_path).convert('L')
    # plt.imshow(mask0, cmap='gray')
    mask = transforms.ToTensor()(mask0)
    src_path = '/home/lix/mySTN11/datasets/try/source.png'
    src0 = Image.open(src_path).convert('RGB')
    src = transforms.ToTensor()(src0)
    tar_path = '/home/lix/mySTN11/datasets/try/target.jpg'
    tar0 = Image.open(tar_path).convert('RGB')
    tar0 = transforms.CenterCrop((176, 176))(tar0)
    tar = transforms.ToTensor()(tar0)

    mask.requires_grad_(True)
    src.requires_grad_(True)
    tar.requires_grad_(True)

    # ==== cuda ==============
    # mask = mask.cuda()
    # src = src.cuda()
    # tar = tar.cuda()

    get_blended = GetPIE.apply
    # get_blended = get_blended.cuda()
    blended = get_blended(src, mask.unsqueeze(0), tar)
    save_data = blended.cpu()
    output_dir = "/home/lix/mySTN11/datasets/try/blended.png"
    save_img = tensor2np_uint8(save_data, convert_image=False)
    plt.imsave(output_dir, save_img)

    blended.backward(torch.ones(blended.shape))
    print('blended pass')

# # 构造mask
# bg = torch.zeros((10, 10), dtype=torch.float)
# fg = torch.ones(4, 4)
# com = bg
# com[3:7, 3:7] = fg
# mask = com
# mask.requires_grad_(True)
# # plt.subplot(221)
# # plt.imshow(mask.data.numpy(), cmap='gray')
# # =========== apply my class ==============
# # get_A = GetA.apply
# # # A = torch.sparse.FloatTensor(16, 16).to_dense()
# # A, omega, omega_yx, ngb_flag = get_A(mask)
# # A.backward(torch.ones(A.shape))
# # print('get A pass')
#
# my_nonzero = MyNonzero.apply
#
# # ============ try ========================
# h, w = mask.shape
# # omega = torch.nonzero(mask)
# omega = my_nonzero(mask)
# # ====== 测试是否可以反向传播 =================
# omega.backward(torch.ones(omega.shape, dtype=torch.long))
# print('omega pass')
#
# # y = torch.reshape(omega[:,0],[omega[:,0].shape[0],1])  # y = np.reshape(omega[0], [omega[0].shape[0], 1])
# # x = torch.reshape(omega[:,1], [omega[:,1].shape[0], 1])  # x = np.reshape(omega[1], [omega[1].shape[0], 1])
# # omega_list = torch.cat([y, x], 1)  # omega_list = np.concatenate([y, x], 1)
#
# ngb_flag = []
# omega_yx = torch.zeros((h, w), dtype=torch.int32)  # omega_yx = np.zeros((h, w), dtype=np.int32)
# for index in range(omega.shape[0]):
#     ## pixel location
#     i, j = omega[index]
#
#     ## create neigbourhoods flag
#     ngb_flag.append([check_existence(mask, i, j + 1),
#                      check_existence(mask, i, j - 1),
#                      check_existence(mask, i + 1, j),
#                      check_existence(mask, i - 1, j), ])
#
#     ## store index to dictionary
#     omega_yx[i][j] = index
# ngb_flag = torch.tensor(ngb_flag, dtype=torch.uint8)
#
# N = omega.shape[0]
# A = torch.sparse.FloatTensor(N, N).to_dense()
# for i in range(N):
#
#     ## progress
#     # progress_bar(i, N-1)
#
#     ## fill 4 or -1
#     ## center
#     A[i, i] = 4
#     id_h, id_w = omega[i]
#
#     ## right
#     if (ngb_flag[i][0]):
#         j = omega_yx[id_h][id_w + 1]
#         A[i, j] = -1
#
#     ## left
#     if (ngb_flag[i][1]):
#         j = omega_yx[id_h][id_w - 1]
#         A[i, j] = -1
#
#     ## bottom
#     if (ngb_flag[i][2]):
#         j = omega_yx[id_h + 1][id_w]
#         A[i, j] = -1
#
#     ## up
#     if (ngb_flag[i][3]):
#         j = omega_yx[id_h - 1][id_w]
#         A[i, j] = -1
#
# A.backward(torch.ones(A.shape))
#
# print('ok')
#
# mask_t = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 10, 10] ; torch.Size([1, 1, 176, 176])
# conv1 = nn.Conv2d(1, 1, 3, 1, 1)
# a = torch.ones((1, 1, 3, 3), dtype=torch.float)
# a[0, 0, 1, 1] = 0
# # from Pytorch STN instance
# # Initialize the weights/bias with identity transformation
# #         self.fc_loc[2].weight.data.zero_()
# #         self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
# conv1.weight.data = a  # or: conv1.weight.data.copy_(a)
# conv1.bias.data.zero_()
# # maxpool1 = nn.MaxPool2d(3, 1, 1)
# my_cap = MyCap.apply
# m1 = conv1(mask_t)  # torch.Size([1, 1, 176, 176])
# plt.subplot(223)
# plt.imshow(m1.squeeze().data.numpy(), cmap='gray')
#
# # contour_t = maxpool1(m1)  # torch.Size([1, 1, 176, 176])
# contour_t = my_cap(m1)
# contour_np = contour_t.squeeze().data.numpy()  # <class 'tuple'>: (176, 176)
# plt.subplot(222)
# plt.imshow(contour_np, cmap='gray')
#
# mask_np = mask_t.squeeze().data.numpy()
# contour = (1 - mask_np) * contour_np
# # plt.subplot(223)
# # plt.imshow(1 - mask_np, cmap='gray')
# plt.subplot(224)
# plt.imshow(contour, cmap='gray')
# # plt.imsave(save_path, mask.numpy(), cmap='gray')
# plt.show()



