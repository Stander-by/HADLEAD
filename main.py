import torch
import numpy as np
import torchvision
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader
from torchvision import transforms
from func import FlattenLayer
from func import train_model

dev = "cuda"

# 初始化
num_inputs = 784
num_outputs = 10
num_hiddens_1 = 256
num_hiddens_2 = 56
# 丢弃概率
drop_prob1 = 0.5
drop_prob2 = 0.5

net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_inputs,num_hiddens_1),
    nn.ReLU(),
    nn.Dropout(drop_prob1),
    nn.Linear(num_hiddens_1,num_hiddens_2),
    nn.ReLU(),
    nn.Dropout(drop_prob2),
    nn.Linear(num_hiddens_2,num_outputs)
)

net.cuda(0)  # 将模型部署到cuda:0上进行加速

for param in net.parameters():
    init.normal_(param,mean=0,std=0.1)

# mini_batch操作
batch_size = 256

# mnist_train和mnist_test都是torch.utils.data.Dataset的子类
mnist_train = torchvision.datasets.FashionMNIST(
    root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(
    root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

# dataloader中已经定义了小批量
train_iter = DataLoader(
    mnist_train,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True
)
test_iter = DataLoader(
    mnist_test,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True
)


# 损失函数
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(),lr=0.1)

num_epochs = 10

train_model(net,train_iter,test_iter,loss,num_epochs,batch_size,optimizer)

