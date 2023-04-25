import trojanvision
import argparse
import os
import re
import time
import math
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from test_exp import load_model_from_path

params = {
    'img_size': 32,
    'nc': 3,
    'nz': 512,
    'ngf': 64,
    'lr': 0.0002,
    'beta1': 0.5,
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
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z):
        middle = self.main(z)
        scaled = (middle + 0.5) * 0.5
        output = F.interpolate(scaled, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return output


def train_DCGAN_models(dataset, forder_path):
    name = dataset.name

    files = [f for f in os.listdir(folder_path) if re.search(r'.+\.pth$', f)]
    files = sorted(files)
    for f in tqdm(files):
        if not f.startswith('resnet'): continue
        model = load_model_from_path(f, folder_path, dataset)
        model.eval()

        netG = Generator(params)
        optimG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

        for data in dataset.loader['train']:
            x = dataset.get_data(data)[0]
            anchors = torch.rand(x.shape, device='cuda')
            inter = torch.rand([len(x), 1, 1, 1], device='cuda')
            nx = anchors * (1 - inter) + x * inter

            print(nx.shape)

            z = model.get_final_fm(nx)
            print(z.shape)

            gx = netG(z.data)
            print(gx.shape)

            loss = torch.square(gx - nx.data).sum((1, 2, 3)).mean()
            print(loss.shape)
            print(loss)
            exit(0)

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
