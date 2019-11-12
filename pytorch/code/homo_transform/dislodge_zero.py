import torch


class DislodgeZero(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input[abs(input) < 1e-1] = torch.tensor(1e-1)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[abs(input) < 1e-1] = 1e1
        return grad_input


if __name__ == '__main__':
    dislodge = DislodgeZero.apply
    a = torch.tensor([1., 2., 1e-8, -1, -1e-8, 0])
    a.requires_grad_(True)
    b = dislodge(a)
    y = b * b
    y.backward(torch.ones(y.shape))

    print('ok')
