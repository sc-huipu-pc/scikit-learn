import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import numpy as np
from matplotlib import pyplot as plt

class ArcSoftmax(nn.Module):
    def __init__(self, cls_num, feature_num):
        super().__init__()
        # x[n,v] · w[v,c]
        self.w = nn.Parameter(torch.randn((feature_num, cls_num)))

    def forward(self, x, s, m):
        # x → x / ||x||
        x_norm = F.normalize(x, dim=1)
        # w → w / ||w||
        w_norm = F.normalize(self.w, dim=0)
        # cosθ = 二范数归一化后的 x·w = (x / ||x||)(w / ||w||)
        # /10：防止梯度爆炸，要在后边乘回来
        cosa = torch.matmul(x_norm, w_norm) / 10
        # 反余弦求角度
        a = torch.acos(cosa)
        # 全部：torch.sum(torch.exp(s * cosa * 10), dim=1, keepdim=True)
        # 当前：torch.exp(s * cosa * 10)
        # 加大角度：torch.exp(s * torch.cos(a + m) * 10)
        arcsoftmax = torch.exp(s * torch.cos(a + m) * 10) / (
                torch.sum(torch.exp(s * cosa * 10), dim=1, keepdim=True) - torch.exp(s * cosa * 10) + torch.exp(s * torch.cos(a + m) * 10))
        return arcsoftmax


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.PReLU(), nn.MaxPool2d(3, 2)
        )
        # 特征
        self.feature_layer = nn.Sequential(
            nn.Linear(11 * 11 * 64, 256), nn.BatchNorm1d(256), nn.PReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.PReLU(),
            nn.Linear(128, 2), nn.PReLU()
        )
        # 分类
        self.arcsoftmax = ArcSoftmax(10, 2)

    def forward(self, x, s, m):
        conv = self.conv_layer(x)
        conv = conv.reshape(x.size(0), -1)
        feature = self.feature_layer(conv)
        out = self.arcsoftmax(feature, s, m)
        out = torch.log(out)
        return feature, out


data_path = r"MNIST"
net_path = r"params/net.pth"
img_path = r"images"

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
dataset = datasets.MNIST(data_path, train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=500, shuffle=True)

net = MyNet()
if os.path.exists(net_path):
    net.load_state_dict(torch.load(net_path))
opt = torch.optim.Adam(net.parameters())
loss_fn = nn.NLLLoss()

# 画图
def visualize(feat, labels, epoch):
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    plt.title("epoch={}".format(epoch))
    plt.show()
    plt.savefig('{}/{}.jpg'.format(img_path, epoch))

if __name__ == '__main__':
    epoch = 0
    while True:
        feat_loader = []
        label_loader = []
        accuracy=0
        num=0
        for i, (x, y) in enumerate(dataloader):
            feature, out = net(x, 1, 1)
            loss = loss_fn(out, y)
            # print(torch.argmax(out,1))
            # print(out)
            # print(y)
            accuracy+=((torch.argmax(out,1)==y).sum()).detach().numpy()
            num+=y.shape[0]
            print("{:0.3f}".format(accuracy/num))
            opt.zero_grad()
            loss.backward()
            opt.step()
            print("epoch:{},i:{},loss:{:.5}".format(epoch, i, loss))
            feat_loader.append(feature)
            label_loader.append((y))
        torch.save(net.state_dict(), net_path)
        features = torch.cat(feat_loader, 0)
        labels = torch.cat(label_loader, 0)
        epoch += 1
        visualize(features.detach().numpy(), labels.detach().numpy(), epoch)
