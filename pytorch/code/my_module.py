import torch
from torch import nn as nn


class MyMultAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weights, bias):
        ctx.save_for_backward(inp, weights, bias)
        return inp*weights+bias

    @staticmethod
    def backward(ctx, grad_output):
        inp, weights, bias = ctx.saved_tensors
        grad_input = grad_output.clone()
        return weights*grad_input, x*grad_input, grad_input


class MyAxAddB(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.mult_add = MyMultAdd.apply

    def forward(self, x):
        return self.mult_add(x, self.weights, self.bias)


class AxAddB(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x*self.weights + self.bias


model = MyAxAddB()
loss_func = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.1)

for i in range(1000):
    x = torch.randn(1)
    model.zero_grad()
    y_pre = model(x)
    loss = loss_func(y_pre, 12*x+6)

    loss.backward()
    opt.step()

print(model.weights, model.bias)

print('ok')
