#!/usr/bin/env python3

import numpy
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

batch_size = 64
learning_rate = 1e-4
num_epoches = 20


def to_img(x):
    x = (x + 1.) * 0.5
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class DCautoencoder(nn.Module):
    def __init__(self):
        super(DCautoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()

        self.fc1 = nn.Linear(784,400)
        self.fc21 = nn.Linear(400,20)
        self.fc22 = nn.Linear(400,20)
        self.fc3 = nn.Linear(20,400)
        self.fc4 = nn.Linear(400,784)

    def encoder(self,x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1),self.fc22(h1)
    
    def reparamentrize(self,mu,logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        
        eps = Variable(eps)

        return eps.mul(std).add_(mu)

    def decoder(self,z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self,x):
        mu,logvar = self.encoder(x)
        z = self.reparamentrize(mu,logvar)
        return self.decoder(z),mu,logvar

data_tf = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

train_dataset = datasets.MNIST(
    root='./data', train=True, transform=data_tf, download=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = autoencoder()
# model = DCautoencoder()
if torch.cuda.is_available():
    model = model.cuda()

# 损失函数
critertion = nn.MSELoss()

# 优化
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoches):
    print('Epoch {}/{} train ok'.format(epoch, num_epoches))
    i = 0

    for data in train_loader:
        img, y_train = data

        img = img.view(img.size(0), -1)
        img = Variable(img)

        outputs = model(img)

        optimizer.zero_grad()
        loss = critertion(outputs, img)

        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print('Epoch {}/{},loss:{:.4f}'.format(epoch, num_epoches, loss.item()))
            pic = to_img(outputs.data)
            save_image(pic, 'image_{}.png'.format(epoch+1))

code = Variable(torch.FloatTensor([[1.19, -3.36, 2.06]]))
decode = model.decoder(code)
decode_img = to_img(decode).squeeze()
decode_img = decode_img.data.numpy()*255
plt.imshow(decode_img.astype('uint8'), cmap='gray')
plt.show()
