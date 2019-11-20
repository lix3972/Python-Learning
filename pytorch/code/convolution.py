# Python和PyTorch对比实现卷积convolution函数及反向传播
# 版权声明：本文为CSDN博主「BrightLampCsdn」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/oBrightLamp/article/details/84589545

import torch
import numpy as np


class Conv2d:
    def __init__(self, stride=1):
        self.weight = None
        self.bias = None

        self.stride = stride

        self.x = None
        self.dw = None
        self.db = None

        self.input_height = None
        self.input_width = None
        self.weight_height = None
        self.weight_width = None
        self.output_height = None
        self.output_width = None

    def __call__(self, x):
        self.x = x
        self.input_height = np.shape(x)[0]
        self.input_width = np.shape(x)[1]
        self.weight_height = np.shape(self.weight)[0]
        self.weight_width = np.shape(self.weight)[1]

        self.output_height = int((self.input_height - self.weight_height) / self.stride) + 1
        self.output_width = int((self.input_width - self.weight_width) / self.stride) + 1

        out = np.zeros((self.output_height, self.output_width))
        for i in range(self.output_height):
            for j in range(self.output_width):
                for r in range(self.weight_height):
                    for s in range(self.weight_width):
                        out[i, j] += x[i * self.stride + r, j * self.stride + s] * self.weight[r, s]
        out = out + self.bias
        return out

    def backward(self, d_loss):
        dx = np.zeros_like(self.x)
        self.dw = np.zeros_like(self.weight)
        self.db = np.zeros_like(self.bias)

        for i in range(self.output_height):
            for j in range(self.output_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.weight_height
                end_j = start_j + self.weight_width
                dx[start_i: end_i, start_j:end_j] += d_loss[i, j] * self.weight

                for u in range(self.weight_height):
                    for v in range(self.weight_width):
                        self.dw[u, v] += d_loss[i, j] * self.x[start_i + u, start_j + v]

        self.db = np.sum(d_loss)

        return dx


np.set_printoptions(precision=8, suppress=True, linewidth=120)
np.random.seed(123)
torch.random.manual_seed(123)

x_numpy = np.random.random((1, 3, 5, 5))
x_tensor = torch.tensor(x_numpy, requires_grad=True)

conv2d_tensor = torch.nn.Conv2d(3, 1, (3, 3), stride=2).double()

conv2d_numpy_channel_0 = Conv2d(stride=2)
conv2d_numpy_channel_0.weight = conv2d_tensor.weight.data.numpy()[0, 0]
conv2d_numpy_channel_0.bias = conv2d_tensor.bias.data.numpy()[0]

conv2d_numpy_channel_1 = Conv2d(stride=2)
conv2d_numpy_channel_1.weight = conv2d_tensor.weight.data.numpy()[0, 1]
conv2d_numpy_channel_1.bias = conv2d_tensor.bias.data.numpy()[0]

conv2d_numpy_channel_2 = Conv2d(stride=2)
conv2d_numpy_channel_2.weight = conv2d_tensor.weight.data.numpy()[0, 2]
conv2d_numpy_channel_2.bias = conv2d_tensor.bias.data.numpy()[0]

out_numpy_0 = conv2d_numpy_channel_0(x_numpy[0, 0])
out_numpy_1 = conv2d_numpy_channel_1(x_numpy[0, 1])
out_numpy_2 = conv2d_numpy_channel_2(x_numpy[0, 2])
out_numpy = out_numpy_0 + out_numpy_1 + out_numpy_2 - conv2d_numpy_channel_0.bias * 2
out_tensor = conv2d_tensor(x_tensor)

d_loss_numpy = np.random.random(out_tensor.shape)
d_loss_tensor = torch.tensor(d_loss_numpy)

dx_numpy_0 = conv2d_numpy_channel_0.backward(d_loss_numpy[0][0])
dx_numpy_1 = conv2d_numpy_channel_1.backward(d_loss_numpy[0][0])
dx_numpy_2 = conv2d_numpy_channel_2.backward(d_loss_numpy[0][0])

out_tensor.backward(d_loss_tensor)
dx_tensor = x_tensor.grad

dw_numpy_0 = conv2d_numpy_channel_0.dw
dw_numpy_1 = conv2d_numpy_channel_1.dw
dw_numpy_2 = conv2d_numpy_channel_2.dw
dw_tensor = conv2d_tensor.weight.grad

db_numpy = conv2d_numpy_channel_0.db
db_tensor = conv2d_tensor.bias.grad

print("out_numpy \n", out_numpy)
print("out_tensor \n", out_tensor.data.numpy())

print("dx_numpy_0 \n", dx_numpy_0)
print("dx_numpy_1 \n", dx_numpy_1)
print("dx_numpy_2 \n", dx_numpy_2)
print("dx_tensor \n", dx_tensor.data.numpy())

print("dw_numpy_0 \n", dw_numpy_0)
print("dw_numpy_1 \n", dw_numpy_1)
print("dw_numpy_2 \n", dw_numpy_2)
print("dw_tensor \n", dw_tensor.data.numpy())

print("db_numpy \n", db_numpy)
print("db_tensor \n", db_tensor.data.numpy()[0])  

