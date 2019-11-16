import torch


if __name__ == '__main__':
    a = torch.tensor([[1., 2.], [3., 4.]])
    a.requires_grad_(True)
    # x = torch.tensor([[-4], [4.5]])
    # x.requires_grad_(True)
    b = torch.tensor([[5.], [6.]])
    b.requires_grad_(True)
    # b_fn = a.mm(x)
    # b_fn.backward(torch.ones(b_fn.shape))

    a_inv = torch.inverse(a)
    x_fn = a_inv.mm(b)
    x_fn.backward(torch.ones(x_fn.shape))

    print('a.grad = ', a.grad)
    print('b.grad = ', b.grad)

    print('ok')
