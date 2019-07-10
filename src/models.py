import torch
from torch import nn


class Discriminator(nn.Module):
  def __init__(self, ndf, image_size, labels):
    super(Discriminator, self).__init__()
    self.label_emb = nn.Embedding(10, 10)
    self.linear1 = nn.Linear(image_size + labels, ndf * 4)
    self.linear2 = nn.Linear(ndf * 4, ndf * 2)
    self.linear3 = nn.Linear(ndf * 2, ndf)
    self.linear4 = nn.Linear(ndf, 1)
    self.leakyReLU = nn.LeakyReLU(0.2, inplace=True)
    self.dropout = nn.Dropout(0.3)

  def forward(self, x, labels):
    c = self.label_emb(labels)
    x = torch.cat([x, c], 1)
    x = self.dropout(self.leakyReLU(self.linear1(x)))
    x = self.dropout(self.leakyReLU(self.linear2(x)))
    x = self.dropout(self.leakyReLU(self.linear3(x)))
    x = torch.sigmoid(x)
    return x


class Generator(nn.Module):
  def __init__(self, nz, ngf, image_size, labels):
    super(Generator, self).__init__()
    self.label_emb = nn.Embedding(10, 10)
    self.linear1 = nn.Linear(nz + labels, ndf)
    self.linear2 = nn.Linear(ndf, ndf * 2)
    self.linear3 = nn.Linear(ndf * 2, ndf * 4)
    self.linear4 = nn.Linear(ndf * 4, image_size + labels)
    self.leakyReLU = nn.LeakyReLU(0.2, inplace=True)

  def forward(self, x, labels):
    c = self.label_emb(labels)
    x = torch.cat([x, c], 1)
    x = self.leakyReLU(self.linear1(x))
    x = self.leakyReLU(self.linear2(x))
    x = self.leakyReLU(self.linear3(x))
    x = torch.tanh(x)
    return x
