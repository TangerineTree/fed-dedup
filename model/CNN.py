from torch import nn
import torch.nn.functional as F
import torch
#两个卷积、两个池化、两个全连接层的卷积神经网络
##1、init函数中要说明输入和输出in_ch out_ch
##2、在forward函数中把各个部分连接起来
##注意如果要用这个model,MNIST和fashionMnist数据的维度要进行reshap为（x,1,28,28）
# #如果要进行CNN神经网络，输入要进行reshap满足这个网络的输入要求,下面这段代码可在dataset.py里
# x_train=x_train.reshape(50000,1,28,28)
# x_valid=x_valid.reshape(10000,1,28,28)
# x_test=x_test.reshape(10000,1,28,28)
#这个针对Mnist和fashionMnist,因为他们是灰色的，没有彩色通道

#根据《Tempered Sigmoid Activations for Deep Learning with Differential Privacy》写的CNN，用于Mnist和fashionMnist数据集
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(1, 16, 8, 2, padding=2),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, 1),
                                      nn.Conv2d(16, 32, 4, 2),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, 1),
                                      nn.Flatten(),
                                      nn.Linear(32 * 4 * 4, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 10))
    def forward(self,x):
        x=self.conv(x)
        return x


#基于文章《Tempered Sigmoid Activations for Deep Learning with Differential Privacy》将激活函数改为Tanh
class CNN_tanh(nn.Module):
    def __init__(self):
        super(CNN_tanh, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(1, 16, 8, 2, padding=2),
                                      nn.Tanh(),
                                      nn.MaxPool2d(2, 1),
                                      nn.Conv2d(16, 32, 4, 2),
                                      nn.Tanh(),
                                      nn.MaxPool2d(2, 1),
                                      nn.Flatten(),
                                      nn.Linear(32 * 4 * 4, 32),
                                      nn.Tanh(),
                                      nn.Linear(32, 10))
    def forward(self,x):
        x=self.conv(x)
        return x


def standardize(x, bn_stats):
    if bn_stats is None:
        return x

    bn_mean, bn_var = bn_stats

    view = [1] * len(x.shape)
    view[1] = -1
    x = (x - bn_mean.view(view)) / torch.sqrt(bn_var.view(view) + 1e-5)

    # if variance is too low, just ignore
    x *= (bn_var.view(view) != 0).float()
    return x

#文章《Large Language Models Can Be Strong Differentially Private Learners》提供的用于Mnist和fashionMnist的模型
#调用方式    centralized_model = CIFAR10_CNN(1, input_norm=None, num_groups=None, size=None)
class MNIST_CNN(nn.Module):
    def __init__(self, in_channels=1, input_norm=None, **kwargs):
        super(MNIST_CNN, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):
        if self.in_channels == 1:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 8, 2, 2), 'M', (ch2, 4, 2, 0), 'M']
            self.norm = nn.Identity()
        else:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 3, 2, 1), (ch2, 3, 1, 1)]
            if input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            elif input_norm == "BN":
                self.norm = lambda x: standardize(x, bn_stats)
            else:
                self.norm = nn.Identity()

        layers = []

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
            else:
                filters, k_size, stride, pad = v
                conv2d = nn.Conv2d(c, filters, kernel_size=k_size, stride=stride, padding=pad)

                layers += [conv2d, nn.Tanh()]
                c = filters

        self.features = nn.Sequential(*layers)

        hidden = 32
        self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden),
                                        nn.Tanh(),
                                        nn.Linear(hidden, 10))

    def forward(self, x):
        if self.in_channels != 1:
            x = self.norm(x.view(-1, self.in_channels, 7, 7))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x