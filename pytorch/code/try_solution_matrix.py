# This file in mySTN10/tryfils/
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
        # grad_input[input > 1] = input
        return grad_input


class MyReLU(torch.autograd.Function):
    """
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html#sphx-glr-beginner-examples-autograd-two-layer-net-custom-function-py
    在ReLU的基础上增加 > 1 输出 1
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


class Multi(torch.autograd.Function):
    """
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html#sphx-glr-beginner-examples-autograd-two-layer-net-custom-function-py
    multify: a * b
    """
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)

        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_a = b * grad_input
        grad_b = a * grad_input
        return grad_a, grad_b


if __name__ == '__main__':
    my_cap = MyCap.apply
    my_relu = MyReLU.apply
    multi = Multi.apply

    a = torch.tensor([1., 2., 1e-8, -1, -1e-8, 0])
    a.requires_grad_(True)
    y = multi(a, a)
    y.backward(torch.ones(y.shape))

    a_relu = torch.tensor([1., 2., 1e-8, -1, -1e-8, 0])
    a_relu.requires_grad_(True)
    b_relu = my_relu(a_relu)
    y_relu = b_relu * b_relu
    y_relu.backward(torch.ones(y_relu.shape))

    a_cap = torch.tensor([1., 2., 1e-8, -1, -1e-8, 0])
    a_cap.requires_grad_(True)
    b_cap = my_cap(a_cap)
    b_cap.requires_grad_(True)
    y_cap = b_cap * b_cap
    y_cap.backward(torch.ones(y_cap.shape))


    # y_relu.backward(torch.ones(y_relu.shape))
    print('a.grad = ', a.grad)
    print('a_relu.grad = ', a_relu.grad)
    print('a_cap.grad = ', a_cap.grad)
    print('ok')
