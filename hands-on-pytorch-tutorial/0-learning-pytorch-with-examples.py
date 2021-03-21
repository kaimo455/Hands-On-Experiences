#!usr/bin/python

##
##
## Numpy
##
##
# import numpy as np
#
# BATCH_SIZE = 64
# DIM_IN = 1000
# HIDDEN_UNIT = 100
# DIM_OUT = 10
#
# # create random input and output
# x = np.random.randn(BATCH_SIZE, DIM_IN)
# y = np.random.randn(BATCH_SIZE, DIM_OUT)
#
# # randomly initialize weights
# w1 = np.random.randn(DIM_IN, HIDDEN_UNIT)
# w2 = np.random.randn(HIDDEN_UNIT, DIM_OUT)
#
# lr = 1e-6
#
# for epoch in range(500):
#     # forward
#     h = x.dot(w1)
#     h_relu = np.maximum(h, 0)
#     y_pred = h_relu.dot(w2)
#     # compute and print loss
#     loss = np.square(y_pred - y).sum()
#     print("epoch {:.4f}, loss {:.4f}".format(epoch, loss))
#     # backprop
#     grad_y_pred = 2 * (y_pred - y)
#     grad_w2 = h_relu.T.dot(grad_y_pred) # h_relu * w2 = y_pred
#     grad_h_relu = grad_y_pred.dot(w2.T) # h_relu * w2 = y_pred
#     grad_h = grad_h_relu.copy()
#     grad_h[h < 0] = 0
#     grad_w1 = x.T.dot(grad_h)
#     # update weight
#     w1 -= lr * grad_w1
#     w2 -= lr * grad_w2

##
##
## Torch without autograd
##
##
# import torch
#
# DTYPE = torch.float
# DEVICE = torch.device('cpu')
#
# BATCH_SIZE = 64
# DIM_IN = 1000
# HIDDEN_UNIT = 100
# DIM_OUT = 10
#
# x = torch.randn(BATCH_SIZE, DIM_IN, device=DEVICE, dtype=DTYPE)
# y = torch.randn(BATCH_SIZE, DIM_OUT, device=DEVICE, dtype=DTYPE)
#
# w1 = torch.randn(DIM_IN, HIDDEN_UNIT, device=DEVICE, dtype=DTYPE)
# w2 = torch.randn(HIDDEN_UNIT, DIM_OUT, device=DEVICE, dtype=DTYPE)
#
# lr = 1e-6
# for epoch in range(500):
#     # forward
#     h = x.mm(w1)
#     h_relu = h.clamp(min=0)
#     y_pred = h_relu.mm(w2)
#     # comput and print loss
#     loss = (y_pred - y).pow(2).sum().item()
#     if epoch % 100 == 0:
#         print("epoch {:.4f}, loss {:.4f}".format(epoch, loss))
#     # backprop
#     grad_y_pred = 2 * (y_pred - y)
#     grad_w2 = h_relu.t().mm(grad_y_pred)
#     grad_h_relu = grad_y_pred.mm(w2.t())
#     grad_h = grad_h_relu.clone()
#     grad_h[h < 0] = 0
#     grad_w1 = x.t().mm(grad_h)
#     # update weights
#     w1 -= lr * grad_w1
#     w2 -= lr * grad_w2

##
##
## Torch with autograd
##
##
# import torch
#
# DTYPE = torch.float
# DEVICE = torch.device('cpu')
#
# BATCH_SIZE = 64
# DIM_IN = 1000
# HIDDEN_UNIT = 100
# DIM_OUT = 10
#
# x = torch.randn(BATCH_SIZE, DIM_IN, device=DEVICE, dtype=DTYPE)
# y = torch.randn(BATCH_SIZE, DIM_OUT, device=DEVICE, dtype=DTYPE)
#
# w1 = torch.randn(DIM_IN, HIDDEN_UNIT, device=DEVICE, dtype=DTYPE, requires_grad=True)
# w2 = torch.randn(HIDDEN_UNIT, DIM_OUT, device=DEVICE, dtype=DTYPE, requires_grad=True)
#
# lr = 1e-6
# for epoch in range(500):
#     y_pred = x.mm(w1).clamp(min=0).mm(w2)
#     loss = (y_pred - y).pow(2).sum()
#     if epoch % 100 == 0:
#         print("epoch {:.4f}, loss {:.4f}".format(epoch, loss.item()))
#     loss.backward()
#     with torch.no_grad():
#         w1 -= lr * w1.grad
#         w2 -= lr * w2.grad
#         w1.grad.zero_()
#         w2.grad.zero_()

##
##
## Torch with customized autograd function
##
##
# import torch
#
# class MyReLU(torch.autograd.Function):
#
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         return input.clamp(min=0)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad_input[input < 0] = 0
#         return grad_input
#
# DTYPE = torch.float
# DEVICE = torch.device('cpu')
#
# BATCH_SIZE = 64
# DIM_IN = 1000
# HIDDEN_UNIT = 100
# DIM_OUT = 10
#
# x = torch.randn(BATCH_SIZE, DIM_IN, device=DEVICE, dtype=DTYPE)
# y = torch.randn(BATCH_SIZE, DIM_OUT, device=DEVICE, dtype=DTYPE)
#
# w1 = torch.randn(DIM_IN, HIDDEN_UNIT, device=DEVICE, dtype=DTYPE, requires_grad=True)
# w2 = torch.randn(HIDDEN_UNIT, DIM_OUT, device=DEVICE, dtype=DTYPE, requires_grad=True)
#
# lr = 1e-6
#
# for epoch in range(500):
#     # To apply our Function, we use Function.apply method. We alias this as 'relu'.
#     relu = MyReLU.apply
#     y_pred = relu(x.mm(w1)).mm(w2)
#     loss = (y_pred - y).pow(2).sum()
#     if epoch % 100 == 0:
#         print("epoch {:.4f}, loss {:.4f}".format(epoch, loss.item()))
#     loss.backward()
#     with torch.no_grad():
#         w1 -= lr * w1.grad
#         w2 -= lr * w2.grad
#         w1.grad.zero_()
#         w2.grad.zero_()

##
##
## Torch with nn module
##
##
# import torch
#
# DTYPE = torch.float
# DEVICE = torch.device('cpu')
#
# BATCH_SIZE = 64
# DIM_IN = 1000
# HIDDEN_UNIT = 100
# DIM_OUT = 10
#
# x = torch.randn(BATCH_SIZE, DIM_IN, device=DEVICE, dtype=DTYPE)
# y = torch.randn(BATCH_SIZE, DIM_OUT, device=DEVICE, dtype=DTYPE)
#
# model = torch.nn.Sequential(
#     torch.nn.Linear(DIM_IN, HIDDEN_UNIT),
#     torch.nn.ReLU(),
#     torch.nn.Linear(HIDDEN_UNIT, DIM_OUT)
# )
#
# loss_fn = torch.nn.MSELoss(reduction='sum')
# lr = 1e-4
# for epoch in range(500):
#     y_pred = model(x)
#     loss = loss_fn(y_pred, y)
#     if epoch % 100 == 0:
#         print("epoch {:.4f}, loss {:.4f}".format(epoch, loss.item()))
#     model.zero_grad()
#     loss.backward()
#     with torch.no_grad():
#         for param in model.parameters():
#             param -= lr * param.grad

##
##
## Torch with optim module
##
##
# import torch
#
# DTYPE = torch.float
# DEVICE = torch.device('cpu')
#
# BATCH_SIZE = 64
# DIM_IN = 1000
# HIDDEN_UNIT = 100
# DIM_OUT = 10
#
# x = torch.randn(BATCH_SIZE, DIM_IN, device=DEVICE, dtype=DTYPE)
# y = torch.randn(BATCH_SIZE, DIM_OUT, device=DEVICE, dtype=DTYPE)
#
# model = torch.nn.Sequential(
#     torch.nn.Linear(DIM_IN, HIDDEN_UNIT),
#     torch.nn.ReLU(),
#     torch.nn.Linear(HIDDEN_UNIT, DIM_OUT)
# )
# loss_fn = torch.nn.MSELoss(reduction='sum')
#
# lr = 1e-4
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
# for epoch in range(500):
#     y_pred = model(x)
#     loss = loss_fn(y_pred, y)
#     if epoch % 100 == 0:
#         print("epoch {:.4f}, loss {:.4f}".format(epoch, loss.item()))
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

##
##
## Torch with customized nn modules
##
##
# import torch
#
# class TwoLayerNet(torch.nn.Module):
#     def __init__(self, dim_in, hidden_unit, dim_out):
#         super(TwoLayerNet, self).__init__()
#         self.linear1 = torch.nn.Linear(dim_in, hidden_unit)
#         self.linear2 = torch.nn.Linear(hidden_unit, dim_out)
#
#     def forward(self, x):
#         h_relu = self.linear1(x).clamp(min=0)
#         y_pred = self.linear2(h_relu)
#         return y_pred
#
# DTYPE = torch.float
# DEVICE = torch.device('cpu')
#
# BATCH_SIZE = 64
# DIM_IN = 1000
# HIDDEN_UNIT = 100
# DIM_OUT = 10
#
# x = torch.randn(BATCH_SIZE, DIM_IN, device=DEVICE, dtype=DTYPE)
# y = torch.randn(BATCH_SIZE, DIM_OUT, device=DEVICE, dtype=DTYPE)
#
# model = TwoLayerNet(DIM_IN, HIDDEN_UNIT, DIM_OUT)
#
# criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#
# for epoch in range(500):
#     y_pred = model(x)
#     loss = criterion(y_pred, y)
#     if epoch % 100 == 0:
#         print("epoch {:.4f}, loss {:.4f}".format(epoch, loss.item()))
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()


##
##
## Torch with control flow + weight sharing
##
##
import torch
import random

class DynamicNet(torch.nn.Module):
    def __init__(self, d_in, h, d_out):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(d_in, h)
        self.middle_linear = torch.nn.Linear(h, h)
        self.output_linear = torch.nn.Linear(h, d_out)

    def forward(self, x):
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0 ,3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred

DTYPE = torch.float
DEVICE = torch.device('cpu')

BATCH_SIZE = 64
DIM_IN = 1000
HIDDEN_UNIT = 100
DIM_OUT = 10

x = torch.randn(BATCH_SIZE, DIM_IN, device=DEVICE, dtype=DTYPE)
y = torch.randn(BATCH_SIZE, DIM_OUT, device=DEVICE, dtype=DTYPE)

model = DynamicNet(DIM_IN, HIDDEN_UNIT, DIM_OUT)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

for epoch in range(500):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    if epoch % 100 == 0:
        print("epoch {:.4f}, loss {:.4f}".format(epoch, loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
