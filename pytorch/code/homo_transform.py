import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch._thnn import type2backend
# from thnn_auto import function_by_name
import torch.backends.cudnn as cudnn


def Homo_grid_generator(theta, size, device):

    N, C, H, W = size
    base_grid = theta.new(N, H, W, 3)  # 创建一个新的数据，数据类型同theta(tensor)

    linear_points = torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1])  # 1行W列 起点-1，终点1(最后一个元素必为1)
    a0 = torch.ger(torch.ones(H), linear_points).expand_as(base_grid[:, :, :, 0])  # torch.ger 外积:
    # ones(H)乘以linear_points第一个元素得第一列,ones(H)乘以linear_points第二个得第二列……
    # 外积结果 H行 W列(linear_points有W列)
    # expand_as(base_grid[:, :, :, 0]) 数据扩展到与base_grid[:, :, :, 0]大小相同：N×H×W

    linear_points = torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])  # 1行H列
    a1 = torch.ger(linear_points, torch.ones(W)).expand_as(base_grid[:, :, :, 1])  # H行W列,扩展到 N×H×W

    a2 = torch.ones(N, H, W, dtype=torch.float)  # N×H×W
    a = torch.stack([a0,a1,a2],3).to(device)  #

    grid = torch.bmm(a.view(N, H * W, 3), theta.transpose(1, 2))
    g1 = grid[:, :, 0] / grid[:, :, 2]
    g2 = grid[:, :, 1] / grid[:, :, 2]
    g = torch.stack([g1, g2], 2)
    grid2 = g.view(N, H, W, 2)

    return grid2

def Homo_grid_generator2(theta, size):
    if theta.data.is_cuda:
        if not cudnn.enabled:
            raise RuntimeError("AffineGridGenerator needs CuDNN for "
                               "processing CUDA inputs, but CuDNN is not enabled")
        if not cudnn.is_acceptable(theta.data):
            raise RuntimeError("AffineGridGenerator generator theta not acceptable for CuDNN")
        N, C, H, W = size
        return torch.cudnn_affine_grid_generator(theta, N, C, H, W)
    else:
        return HomoGridGenerator.apply(theta, size)

# TODO: Port these completely into C++


class HomoGridGenerator(Function):

    @staticmethod
    def _enforce_cudnn(input):
        if not cudnn.enabled:
            raise RuntimeError("AffineGridGenerator needs CuDNN for "
                               "processing CUDA inputs, but CuDNN is not enabled")
        assert cudnn.is_acceptable(input)

    @staticmethod
    def forward(ctx, theta, size):
        assert type(size) == torch.Size
        N, C, H, W = size
        ctx.size = size
        if theta.is_cuda:
            HomoGridGenerator._enforce_cudnn(theta)
            assert False
        ctx.is_cuda = False
        base_grid = theta.new(N, H, W, 3)
        linear_points = torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1])
        base_grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(base_grid[:, :, :, 0])
        linear_points = torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])
        base_grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(base_grid[:, :, :, 1])
        base_grid[:, :, :, 2] = 1
        ctx.base_grid = base_grid
        grid = torch.bmm(base_grid.view(N, H * W, 3), theta.transpose(1, 2))
        grid[:, :, 0] = grid[:, :, 0]/grid[:, :, 2]
        grid[:, :, 1] = grid[:, :, 1]/grid[:, :, 2]
        grid = grid.view(N, H, W, 3)[:, :, :, :2]

        return grid

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_grid):
        N, C, H, W = ctx.size
        assert grad_grid.size() == torch.Size([N, H, W, 3])
        assert ctx.is_cuda == grad_grid.is_cuda
        if grad_grid.is_cuda:
            HomoGridGenerator._enforce_cudnn(grad_grid)
            assert False
        base_grid = ctx.base_grid
        grad_theta = torch.bmm(
            base_grid.view(N, H * W, 3).transpose(1, 2),
            grad_grid.view(N, H * W, 3))
        grad_theta = grad_theta.transpose(1, 2)
        return grad_theta, None

if __name__ == '__main__':
    # grid = F.affine_grid(theta, x.size())
    # x = F.grid_sample(x, grid)
    theta = torch.randn(2,3,3)
    x = torch.randn(2,3,20,30)
    grid = Homo_grid_generator(theta,x.size())

    y = F.grid_sample(x,grid)

    print('ok')
