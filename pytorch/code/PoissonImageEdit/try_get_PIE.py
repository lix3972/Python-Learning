from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from utils.image_show import plt_tensor, tensor2np_uint8
from global_varialbe import *
import time

# truncation n. 切掉顶端, 截头, 截短 【计】 截断; 截除 | truncate vt. 切去头端, 缩短,
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
    if 0 <= id_h <= h - 1 and 0 <= id_w <= w - 1:
        if mask[id_h][id_w] == 1:
            return torch.tensor(1, dtype=torch.uint8)

        else:
            return torch.tensor(0, dtype=torch.uint8)
    else:
        return torch.tensor(0, dtype=torch.uint8)


# =============================================================================
def lap_at_index(source, index, contour, ngb_flag):
    ## current location
    i, j = index
    ## take laplacian
    # N = torch.sum(ngb_flag == True)
    N = ngb_flag.sum()
    # val = (N * source[i, j]
    #        - (float(ngb_flag[0] == True) * source[i, j + 1])
    #        - (float(ngb_flag[1] == True) * source[i, j - 1])
    #        - (float(ngb_flag[2] == True) * source[i + 1, j])
    #        - (float(ngb_flag[3] == True) * source[i - 1, j]))
    val = (N * source[i, j]
            - ngb_flag[0] * source[i, j+1]
            - ngb_flag[1] * source[i, j-1]
            - ngb_flag[2] * source[i+1, j]
            - ngb_flag[3] * source[i-1, j])
    return val


def constrain(target, index, contour, ngb_flag):
    ## current location
    i, j = index
    ## In order to use "Dirichlet boundary condition",
    ## if on boundry, add in target intensity --> set constraint grad(source) = target at boundary
    val_right = (1-ngb_flag[0]) * target[i, j + 1]
    val_left = (1-ngb_flag[1]) * target[i, j - 1]
    val_down = (1-ngb_flag[2]) * target[i + 1, j]
    val_up = (1-ngb_flag[3]) * target[i - 1, j]
    val = val_right + val_left + val_down + val_up
    return val
    # ## If not on boundry, just take grad.
    # else:
    #     val = torch.tensor(0.0)
    #     return val


def importing_gradients(src, tar, omega, contour, ngb_flag, device):
    ### output array
    u_r = torch.zeros(omega.shape[0], device=device)
    u_g = torch.zeros(omega.shape[0], device=device)
    u_b = torch.zeros(omega.shape[0], device=device)
    ### take laplacian

    for index in range(omega.shape[0]):
        ## apply each color channel
        u_r[index] = lap_at_index(src[:, :, 0], omega[index], contour, ngb_flag[index])
        u_g[index] = lap_at_index(src[:, :, 1], omega[index], contour, ngb_flag[index])
        u_b[index] = lap_at_index(src[:, :, 2], omega[index], contour, ngb_flag[index])
        i, j = omega[index]
        if contour[i][j]:
            u_r[index] += constrain(tar[:, :, 0], omega[index], contour, ngb_flag[index])
            u_g[index] += constrain(tar[:, :, 1], omega[index], contour, ngb_flag[index])
            u_b[index] += constrain(tar[:, :, 2], omega[index], contour, ngb_flag[index])
            # u_r[index] = lap_at_index(src[:, :, 0], omega[index], contour, ngb_flag[index]) \
        #               + constrain(tar[:, :, 0], omega[index], contour, ngb_flag[index])
        # u_g[index] = lap_at_index(src[:, :, 1], omega[index], contour, ngb_flag[index]) \
        #               + constrain(tar[:, :, 1], omega[index], contour, ngb_flag[index])
        # u_b[index] = lap_at_index(src[:, :, 2], omega[index], contour, ngb_flag[index]) \
        #               + constrain(tar[:, :, 2], omega[index], contour, ngb_flag[index])

    return u_r, u_g, u_b


def lap_at_index_mixing(source, target, index, contuor, ngb_flag, device):
    # current location
    i, j = index
    # gradient for source image
    grad_right_src = ngb_flag[0] * (source[i, j] - source[i, j + 1])
    grad_left_src = ngb_flag[1] * (source[i, j] - source[i, j - 1])
    grad_bottom_src = ngb_flag[2] * (source[i, j] - source[i + 1, j])
    grad_up_src = ngb_flag[3] * (source[i, j] - source[i - 1, j])

    # gradient for target image
    grad_right_tar = ngb_flag[0] * (target[i, j] - target[i, j + 1])
    grad_left_tar = ngb_flag[1] * (target[i, j] - target[i, j - 1])
    grad_bottom_tar = ngb_flag[2] * (target[i, j] - target[i + 1, j])
    grad_up_tar = ngb_flag[3] * (target[i, j] - target[i - 1, j])
    val = [grad_right_src, grad_left_src, grad_bottom_src, grad_up_src]

    if grad_right_src.abs() < grad_right_tar.abs():
        val[0] = grad_right_tar

    if grad_left_src.abs() < grad_left_tar.abs():
        val[1] = grad_left_tar

    if grad_bottom_src.abs() < grad_bottom_tar.abs():
        val[2] = grad_bottom_tar

    if grad_up_src.abs() < grad_up_tar.abs():
        val[3] = grad_up_tar

    return val[0] + val[1] + val[2] + val[3]


def mixing_gradients(src, tar, omega, contour, ngb_flag, device):
    # output array
    u_r = torch.zeros(omega.shape[0], device=device)
    u_g = torch.zeros(omega.shape[0], device=device)
    u_b = torch.zeros(omega.shape[0], device=device)

    # take laplacian
    for index in range(omega.shape[0]):
        u_r[index] = lap_at_index(src[:, :, 0], omega[index], contour, ngb_flag[index])
        u_g[index] = lap_at_index(src[:, :, 1], omega[index], contour, ngb_flag[index])
        u_b[index] = lap_at_index(src[:, :, 2], omega[index], contour, ngb_flag[index])
        i, j = omega[index]
        if contour[i][j]:
            u_r[index] += constrain(tar[:, :, 0], omega[index], contour, ngb_flag[index])
            u_g[index] += constrain(tar[:, :, 1], omega[index], contour, ngb_flag[index])
            u_b[index] += constrain(tar[:, :, 2], omega[index], contour, ngb_flag[index])

    return u_r, u_g, u_b


class GetPIE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, mask, tar, mask_init):  # mask size N, C, H, W
        # device = torch.device("cuda:{}".format(gpu))
        start_PIE = time.time()
        device = mask.device
        ctx.save_for_backward(src, mask, tar, mask_init)

        # method = 'import'
        method = 'mix'

        if torch.sum(mask) < 10:
            return tar

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
        my_truncate = MyTruncate.apply
        mask[mask > 0] = 1
        mask[mask <= 0] = 0
        # 去除mask中logo到边缘的问题
        mask[:, :, 0, :] = 0
        mask[:, :, :, 0] = 0
        mask[:, :, 175, :] = 0
        mask[:, :, :, 175] = 0
        # CUDA
        conv1 = conv1.to(device)
        mask = mask.to(device)

        m1 = conv1(mask)  # torch.Size([1, 1, 176, 176])
        erode_t = my_truncate(m1)

        # ==== mask = mask.squeeze() ================
        mask = mask.squeeze()
        src = src.squeeze().permute(1, 2, 0)
        tar = tar.squeeze().permute(1, 2, 0)

        # mean = torch.tensor(IMAGE_NORMALIZE_MEAN, device=device)
        # std = torch.tensor(IMAGE_NORMALIZE_STD, device=device)
        # src = std * src + mean
        # tar = std * tar + mean
        # src = src.clamp(min=0, max=1)
        # tar = tar.clamp(min=0, max=1)
        # contour = (1 - mask) * erode_t.squeeze()
        contour = mask * erode_t.squeeze()
        omega = torch.nonzero(mask)

        start_omega_yx = time.time()
        print('contour and omega run time: ', start_omega_yx-start_PIE)
        ngb_flag = []
        omega_yx = torch.zeros((h, w), dtype=torch.int32, device=device)  # omega_yx = np.zeros((h, w), dtype=np.int32)
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
        ngb_flag = torch.tensor(ngb_flag, dtype=torch.uint8, device=device)

        start_A = time.time()
        print('omega_yx run time: ', start_A - start_omega_yx)
        N = omega.shape[0]
        A = torch.sparse.FloatTensor(N, N).to_dense().to(device)
        for i in range(N):
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

        end_A = time.time()
        print('create A run time: ', end_A-start_A)
        ## select process type
        if method == "import":
            start_import_grad = time.time()
            u_r, u_g, u_b = importing_gradients(src, tar, omega, contour, ngb_flag, device)
            end_import_grad = time.time()
            print('import_grad run time: ', end_import_grad - start_import_grad)
        if method == "mix":
            start_mix_grad = time.time()
            u_r, u_g, u_b = mixing_gradients(src, tar, omega, contour, ngb_flag, device)
            end_mix_grad = time.time()
            print('mix_grad run time: ', end_mix_grad-start_mix_grad)

        start_invA = time.time()
        Ainv = torch.inverse(A)
        end_invA = time.time()
        print('invA run time: ', end_invA - start_invA)
        x_r = u_r.unsqueeze(0).mm(Ainv)
        x_g = u_g.unsqueeze(0).mm(Ainv)
        x_b = u_b.unsqueeze(0).mm(Ainv)
        end_xrgb = time.time()
        print('end_xrgb run time: ', end_xrgb-end_invA)
        x_r, x_g, x_b = x_r.squeeze(0), x_g.squeeze(0), x_b.squeeze(0)

        blended = tar.clone()
        overlapped = tar.clone()

        for index in range(omega.shape[0]):
            i, j = omega[index]
            ## normal
            blended[i][j][0] = torch.clamp(x_r[index], 0.0, 1.0)
            blended[i][j][1] = torch.clamp(x_g[index], 0.0, 1.0)
            blended[i][j][2] = torch.clamp(x_b[index], 0.0, 1.0)

            # ## overlapping
            # overlapped[i][j][0] = src[i][j][0]
            # overlapped[i][j][1] = src[i][j][1]
            # overlapped[i][j][2] = src[i][j][2]
        # img_comp = transforms.Normalize(IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD)(blended.permute(2, 0, 1))
        img_comp = blended.permute(2, 0, 1)
        end_comp = time.time()
        print('end_comp run tine: ', end_comp-end_xrgb)
        return img_comp.unsqueeze(0)  # * 255, overlapped * 255

    @staticmethod
    def backward(ctx, grad_output):
        src, mask, tar, mask_init = ctx.saved_tensors
        # device = torch.device("cuda:{}".format(gpu))
        device = mask.device
        l1_grad = nn.L1Loss(reduction='sum')(grad_output, torch.zeros(grad_output.shape, device=device))
        # if torch.sum(mask) > torch.sum(mask_init)/3 and torch.sum(mask) < 1.5 * torch.sum(mask_init):
        #     beta = torch.tensor(1, device=device)
        # else:
        #     # torch.abs(torch.log((torch.sum(mask) + 1e-8) / torch.sum(mask_init))).to(device)
        #     beta = torch.pow(torch.sum(mask)-torch.sum(mask_init), 2)

        grad_stc = l1_grad*torch.ones(src.shape, device=device) / torch.sum(mask)
        grad_mask = l1_grad*torch.ones(mask.shape, device=device) / torch.sum(mask)
        grad_tar = l1_grad*torch.ones(tar.shape, device=device) / torch.sum(torch.ones(tar.shape, device=device))
        tmp_m = mask.repeat(1, 3, 1, 1)
        grad_stc[tmp_m <= 0] = 0
        grad_mask[mask <= 0] = 0

        return grad_stc, grad_mask, grad_tar, None


if __name__ == '__main__':
    start_program = time.time()
    method = 'mix'  # import, mix
    # method = 'import'

    pic_name = 'lining'
    output_dir = '../datasets/try/' + pic_name + '176_blended.png'
    mask_path = '../datasets/try/' + pic_name + '176_mask.png'
    src_path = '../datasets/try/' + pic_name + '176.png'
    tar_path = '../datasets/try/target.jpg'

    transform_logo = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD)])

    mask0 = Image.open(mask_path).convert('L')
    # plt.imshow(mask0, cmap='gray')
    mask = transforms.ToTensor()(mask0)
    src0 = Image.open(src_path).convert('RGB')
    src = transform_logo(src0)
    tar0 = Image.open(tar_path).convert('RGB')
    tar0 = transforms.CenterCrop((176, 176))(tar0)
    tar = transform_logo(tar0)

    mask.requires_grad_(True)
    src.requires_grad_(True)
    tar.requires_grad_(True)

    # ==== cuda ==============
    gpu = torch.tensor(0)
    device = torch.device("cuda:{}".format(gpu))
    mask = mask.unsqueeze(0).to(device)
    src = src.unsqueeze(0).to(device)
    tar = tar.unsqueeze(0).to(device)

    get_blended = GetPIE.apply
    # get_blended = get_blended.cuda()
    start_blended = time.time()
    blended = get_blended(src, mask, tar, mask)
    end_blended = time.time()
    print('blended run time: {}'.format(end_blended - start_blended))
    save_data = blended.squeeze().cpu()
    print('forward run time:', end_blended - start_program)

    save_img = tensor2np_uint8(save_data, convert_image=True)
    plt.imsave(output_dir, save_img)

    blended.backward(torch.ones(blended.shape, device=device))
    end_program = time.time()
    print('program run time: {}'.format(end_program - start_program))
    print('blended pass')
