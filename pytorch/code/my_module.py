import torch
from torch import nn as nn


class AxAddB(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x*self.weights + self.bias


model = AxAddB()
loss_func = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.1)

for i in range(1000):
    x = torch.randn(1)
    model.zero_grad()
    y_pre = model(x)
    loss = loss_func(y_pre, 5*x+6)

    loss.backward()
    opt.step()

print(model.weights, model.bias)

print('ok')
