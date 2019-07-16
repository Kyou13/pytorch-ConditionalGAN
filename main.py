import os
import torch
import torchvision
import torch.nn as nn
import click
import datetime
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from src import models, utils
import matplotlib.pyplot as plt

params = {
    'seed': 123456789,
    'batch_size': 64,  # 大きすぎると学習が進まない
    'optimizer': 'adam',
    'lr': 2e-4,
    'wd': 0,
    'labels': 10,
    'image_size': 784,
    'beta1': 0.5,
    'epochs': 30,
    'latent_size': 64,
    'nz': 100,
    'ndf': 256,
    'ngf': 64,
    'imageSize': 64
}


@click.group()
def main():
  np.random.seed(params['seed'])
  torch.manual_seed(params['seed'])
  torch.cuda.manual_seed_all(params['seed'])
  torch.backends.cudnn.benchmark = True


@main.command()
@click.option('--dataset', type=str, default='mnist')
@click.option('--data', type=str, default='data')
def train(dataset, data):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  sample_dir = os.path.join('samples', dataset)
  weights_dir = os.path.join('weights', dataset)
  os.makedirs(sample_dir, exist_ok=True)
  os.makedirs(weights_dir, exist_ok=True)

  if dataset == 'mnist':
    dataset = torchvision.datasets.MNIST(root=data, download=True, train=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize(  # [0,1] -> [-1, 1]
                                                 (0.5,), (0.5,)),
                                         ]))
    params['nc'] = 1

  data_loader = torch.utils.data.DataLoader(
      dataset=dataset,
      batch_size=params['batch_size'],
      shuffle=True
  )

  D = models.Discriminator(
      params['ndf'], params['image_size'], params['labels'])
  G = models.Generator(params['nz'], params['ngf'],
                       params['image_size'], params['labels'])

  D = D.to(device)
  G = G.to(device)

  criterion = nn.BCELoss()
  d_optimizer = utils.get_optim(params, D)
  g_optimizer = utils.get_optim(params, G)

  d_losses = []
  g_losses = []
  total_step = len(data_loader)
  for epoch in range(params['epochs']):
    for i, (images, labels) in enumerate(data_loader):
      b_size = images.size(0)
      images = images.reshape(b_size, -1).to(device)
      labels = labels.to(device)

      real_labels = torch.ones(b_size).to(device)
      fake_labels = torch.zeros(b_size).to(device)
      random_labels = torch.LongTensor(
          np.random.randint(0, 10, b_size)).to(device)

      # Train discriminator
      outputs = D(images, labels)
      d_loss_real = criterion(outputs, real_labels)
      real_score = outputs

      z = torch.randn(b_size, params['nz']).to(device)
      fake_images = G(z, random_labels)
      outputs = D(fake_images.detach(), random_labels)
      d_loss_fake = criterion(outputs, fake_labels)
      fake_score = outputs

      d_loss = d_loss_real + d_loss_fake
      d_optimizer.zero_grad()
      d_loss.backward()
      d_optimizer.step()

      # Train generator
      # ランダムなラベルを再定義
      random_labels = torch.LongTensor(
          np.random.randint(0, 10, b_size)).to(device)
      fake_images = G(z, random_labels)
      outputs = D(fake_images, random_labels)

      g_loss = criterion(outputs, real_labels)

      g_optimizer.zero_grad()
      g_loss.backward()
      g_optimizer.step()

      print('Epoch [{}/{}], step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
            .format(epoch, params['epochs'], i + 1, total_step, d_loss.item(), g_loss.item(),
                    real_score.mean().item(), fake_score.mean().item()))  # .item():ゼロ次元Tensorから値を取り出す

      g_losses.append(g_loss.item())
      d_losses.append(d_loss.item())

    if (epoch + 1) == 1:
      images = images.reshape(b_size, 1, 28, 28)
      save_image(utils.denorm(images), os.path.join(
          sample_dir, 'real_images.png'))
    fake_images = fake_images.reshape(b_size, 1, 28, 28)
    save_image(utils.denorm(fake_images), os.path.join(
        sample_dir, 'fake_images-{}.png'.format(epoch + 1)))

  torch.save(G.state_dict(), os.path.join(weights_dir, 'G.ckpt'))
  torch.save(D.state_dict(), os.path.join(weights_dir, 'D.ckpt'))

  plt.figure(figsize=(10, 5))
  plt.title("Generator and Discriminator Loss During Training")
  plt.plot(g_losses, label="Generator")
  plt.plot(d_losses, label="Discriminator")
  plt.xlabel("iterations")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig(os.path.join(sample_dir, 'loss.png'))


@main.command()
@click.option('--dataset', type=str, default='mnist')
def generate(dataset):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  sample_dir = os.path.join('samples', dataset)
  weights_dir = os.path.join('weights', dataset)
  os.makedirs(sample_dir, exist_ok=True)

  G = models.Generator(params['nz'], params['ngf'],
                       params['image_size'], params['labels'])
  G.load_state_dict(torch.load(os.path.join(weights_dir, 'G.ckpt')))
  G.eval()
  G = G.to(device)

  with torch.no_grad():
    random_labels = torch.LongTensor(
        np.random.randint(0, 10, params['batch_size'])).to(device)
    z = torch.randn(params['batch_size'], params['nz']).to(device)
    fake_images = G(z, random_labels)

  dt_now = datetime.datetime.now()
  now_str = dt_now.strftime('%y%m%d%H%M%S')
  fake_images = fake_images.reshape(-1, 1, 28, 28)
  save_image(utils.denorm(fake_images), os.path.join(
      sample_dir, 'fake_images_{}.png'.format(now_str)))
  print('Saved Image ' + os.path.join(sample_dir,
                                      'fake_images_{}.png'.format(now_str)))


if __name__ == '__main__':
  main()
