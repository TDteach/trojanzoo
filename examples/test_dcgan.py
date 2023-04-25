import trojanvision
import argparse
import os
import re
import time
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from test_exp import load_model_from_path

params = {
    'img_size': 32,
    'nc': 3,
    'nz': 512,
    'ngf': 64,
    'lr': 0.0002,
    'beta1': 0.5,
    'n_epochs': 100,
}


class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.img_size = params['img_size']
        self.nc = params['nc']
        self.nz = params['nz']
        self.ngf = params['ngf']
        ngf = self.ngf
        nz = self.nz
        nc = self.nc
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            # nn.ReLU(True),
            # nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z):
        middle = self.main(z)
        x = F.avg_pool2d(middle, kernel_size=2)
        scaled = (x.tanh() + 1.0) * 0.5
        return scaled


def train_DCGAN_models(dataset, forder_path):
    name = dataset.name

    files = [f for f in os.listdir(folder_path) if re.search(r'.+\.pth$', f)]
    files = sorted(files)
    for f in tqdm(files):
        if not f.startswith('resnet'): continue
        model = load_model_from_path(f, folder_path, dataset, kwargs)
        model.eval()

        netG = Generator(params)
        optimG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

        netG.cuda()
        netG.train()

        bar = tqdm(range(params['n_epochs']))
        for epoch in bar:
            loss_ema = None
            for data in dataset.loader['train']:
                x = dataset.get_data(data)[0]

                gamma = np.random.beta(0.5, 0.5, x.size(0))
                gamma_tensor = torch.from_numpy(gamma).float().cuda().reshape(-1,1,1,1)
                indices = torch.randperm(x.size(0), device='cuda', dtype=torch.long)
                perm_x = x[indices]
                nx = x * gamma_tensor + perm_x * (1-gamma_tensor)

                '''
                anchors = torch.rand(x.shape, device='cuda')
                inter = torch.rand([len(x), 1, 1, 1], device='cuda')
                nx = anchors * (1 - inter) + x * inter
                '''

                z = model.get_final_fm(nx)
                z_in = z.view(z.shape[0], z.shape[1], 1, 1)

                gx = netG(z_in.data)

                loss = torch.square(gx - nx.data).sum((1, 2, 3)).mean()
                lv = loss.item()
                if loss_ema is None:
                    loss_ema = lv
                else:
                    loss_ema = 0.95*loss_ema + 0.05*lv

                bar.set_description(f'epoch {epoch}: loss={lv:.3f}')

                optimG.zero_grad()
                loss.backward()
                optimG.step()
        exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    kwargs = parser.parse_args().__dict__

    env = trojanvision.environ.create(**kwargs)

    # '''
    dataset = trojanvision.datasets.create(**kwargs)
    print('train DCGAN models for', dataset.name)
    name = dataset.name
    folder_path = f'./benign_{name}'
    train_DCGAN_models(dataset, folder_path)
    # '''

    dataset = trojanvision.datasets.create(**kwargs)

    netG = Generator(params)
    noise = torch.randn(1, params['nz'], 1, 1)
    o = netG(noise)
    print(o.shape)
