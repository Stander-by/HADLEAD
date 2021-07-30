import torchvision
from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader


def load_data_fashion_mnist(batch_size):
    # mnist_train和mnist_test都是torch.utils.data.Dataset的子类
    mnist_train = torchvision.datasets.FashionMNIST(
        root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(
        root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())
    # dataloader中已经定义了小批量
    train_iter = DataLoader(mnist_train,batch_size=batch_size,shuffle=True,pin_memory=True)
    test_iter = DataLoader(mnist_test,batch_size=batch_size,shuffle=True,pin_memory=True)
    return train_iter,test_iter


class LinearNet(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs,num_outputs)

    def forward(self,x):  # x.shape=(batch,1,28,28)
        y = self.linear(x.view(x.shape[0],-1))
        return y

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr*param.grad / batch_size

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            #梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()

            l.backward(retain_graph=True)
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and param[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward(retain_graph=True)
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy1(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (epoch + 1, train_l_sum/n, train_acc_sum/n, test_acc))





class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self,x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


def evaluate_accuracy(data_iter,net):  # 参数net实际上是一个变量
    acc_num = 0.0
    it = 0
    for x,y in data_iter:
        x = x.cuda()
        y = y.cuda()
        acc = (net(x).argmax(dim=1) == y).float().mean().item()
        # print("acc:" + str(acc))
        acc_num += acc
        it += 1

    return acc_num/float(it)

def evaluate_accuracy1(data_iter, net):
    acc_num, n = 0.0, 0
    for X, y in data_iter:
        acc_num += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_num/n



def train_model(net,train_iter,test_iter,loss,num_epoch,batch_size,optimizer):
    """
    训练模型
    :param net:
    :param train_iter:
    :param test_iter:
    :param loss:
    :param num_epoch:
    :param batch_size:
    :param optimizer:
    :return:
    """
    train_loss_list = []
    for epoch in range(num_epoch):
        train_loss_sum, train_acc_sum, n, it = 0.0,0.0,0,0
        for x,y in train_iter:
            x = x.cuda(0)
            y = y.cuda(0)
            y_hat = net(x)  # 模型预测值
            l = loss(y_hat,y).sum()  # 损失函数值
            train_loss_list.append(l.item())

            # 梯度清零
            optimizer.zero_grad()
            # 求梯度
            l.backward()
            # 优化(模型优化更新)
            optimizer.step()

            train_loss_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().mean().item()
            n += y.shape[0]
            it += 1
        test_acc = evaluate_accuracy(test_iter,net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_loss_sum / n, train_acc_sum / it, test_acc))

    x = [i for i in range(len(train_loss_list))]
    plt.scatter(x,train_loss_list,s=2)
    plt.xlabel('iteration')
    plt.ylabel('train_loss')
    plt.show()


# 每次预读取数据到GPU上进行读取加速
class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        # with torch.cuda.stream(self.stream):
        #     self.next_data = self.next_data.cuda(non_blocking=True)

    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data


# 没卵用
def train_model_read_accelerate(net,train_p,test_p,loss,num_epochs,batch_size,optimizer):
    """
    预读取到gpu上进行处理
    :param net:
    :param train_p:
    :param test_p:
    :param loss:
    :param num_epochs:
    :param optimizer:
    :return:
    """
    # 可视化数据准备
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    # 训练
    for epoch in range(num_epochs):
        i = 0  # 记数
        train_loss = 0.0
        train_acc = 0.0
        data = train_p.next()
        while data is not None:
            x, y = data
            x = x.cuda(0)
            y = y.cuda(0)
            # 预测结果
            y_hat = net(x)
            # 损失函数值(求mini_batch的平均)
            l = loss(y_hat, y).sum()/batch_size
            # 梯度清零
            optimizer.zero_grad()
            # 求梯度
            l.backward()
            # 优化
            optimizer.step()
            # 操作聚合
            i += 1
            train_loss += l.item()
            train_acc += (y_hat.argmax(dim=1) == y).float().mean().item()  # .mean()代表求平均了
        # 每个epoch进行测试
        """train"""
        train_acc_list.append(train_acc/float(i))
        train_loss_list.append(train_loss/float(i))
        """test"""
        test_data = data_prefetcher(test_p)
        j = 0
        test_acc = 0.0
        while test_data is not None:
            t,label = test_data
            t = t.cuda()
            label = label.cuda(0)
            t_p = net(t)
            test_acc += (t_p.argmax(dim=1) == label).float().mean().item()
        test_acc_list.append(test_data/float(j))
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_loss/float(i), train_acc/float(i), test_data/float(j)))
        # 作图
        x = [i for i in range(num_epochs)]
        plt.scatter(x,train_loss_list,s=3,label='train_loss')
        plt.plot(x,train_acc_list,label='train_acc')
        plt.plot(x,test_acc_list,label='test_acc')
        plt.show()


def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < keep_prob).float()  # 原本是bool型数组转化成float即是0.0或1.0数组

    return mask * X / keep_prob


class MySequential(nn.Module):
    """
    手写实现Sequential的部分功能
    """
    def __init__(self,*args):
        super(MySequential, self).__init__()
        # 如果传入的是一个OrderedDict
        if len(args) == 1 and isinstance(args[0],OrderedDict):
            for key,val_module in args[0].items():  # .items()将key和value作为一个元组构成的列表返回
                self.add_module(key,val_module)
        else:
            for idx,module in enumerate(args):
                self.add_module(str(idx),module)

    def forward(self,input):
        # self._modules返回一个OrderedDict,保证会按照成员添加时的顺序遍历
        for module in self._modules.values():
            input = module(input)

        return input


class MyDense(nn.Module):
    def __init__(self):
        super(MyDense, self).__init__()
        self.params = nn.ParameterList([
            nn.parameter(torch.randn(4,4) for i in range(3))
        ])
        self.params.append(torch.randn(4,1))

    def forward(self,x):
        for i in range(len(self.params)):
            x = torch.mm(x,self.params)
        return x


class MyDictDence(nn.Module):
    def __init__(self):
        super(MyDictDence, self).__init__()
        self.params = nn.ParameterDict({
            'linear_1':nn.Parameter(torch.randn(4,4)),
            'linear_2':nn.Parameter(torch.randn(4,1))
        })
        # update()新增参数
        self.params.update({'linear_3':nn.Parameter(torch.randn(4,2))})
        # keys()返回所有键值
        # items()返回所有键值对

    def forward(self,x,choice='linear_1'):
        return torch.mm(x,self.params[choice])


class FancyMLP(nn.Module):
    """
    自己创建的类(继承nn.Module具备灵活性)
    """
    def __init__(self,**kwargs):
        """
        :param kwargs:**kwargs表示接受一个键值参数字典
        """
        # 不可训练参数(常熟参数)
        self.rand_weight = torch.rand((20,20),requires_grad=False)
        self.linear = nn.Linear(20,20)

    def forward(self,x):
        x = self.linear(x)
        x = torch.relu(torch.mm(x,self.rand_weight.data) + 1)

        # 复用全连接层。等价于两个全连接层共享参数
        x = self.linear(x)
        # 控制流，这里我们需要调用item函数来返回标量进行比较
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()


def corr2d(x,k):
    h,w = k.shape  # h:height;w:weight
    y = torch.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i,j] = (x[i:i+h,j:j+w] * k).sum()
    return y