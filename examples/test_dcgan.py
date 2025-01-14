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
import torchvision.transforms as transforms

import pickle
import numpy as np

from test_exp import load_model_from_path

from gsw.gswnn import GSW_NN

params = {
    'img_size': 32,
    'nc': 3,
    'nz': 512,
    'ngf': 64,
    'lr': 0.0002,
    'beta1': 0.5,
    'rand_beta': 0.2,
    'n_epochs': 100,
}


class ResidualBlock(nn.Module):
    """A simple residual block."""

    def __init__(self, n_channels, hidden_channels):
        """Initializes a new ResidualBlock instance.
        Args:
            n_channels: Number of input and output channels.
            hidden_channels: Number of hidden channels.
        """
        super().__init__()
        self._net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_channels, out_channels=n_channels, kernel_size=1
            ),
        )

    def forward(self, x):
        return x + self._net(x)


class ResidualStack(nn.Module):
    """A stack of multiple ResidualBlocks."""

    def __init__(self, n_channels, hidden_channels, n_residual_blocks=1):
        """Initializes a new ResidualStack instance.
        Args:
            n_channels: Number of input and output channels.
            hidden_channels: Number of hidden channels.
            n_residual_blocks: Number of residual blocks in the stack.
        """
        super().__init__()
        self._net = nn.Sequential(
            *[
                 ResidualBlock(n_channels, hidden_channels)
                 for _ in range(n_residual_blocks)
             ]
             + [nn.ReLU()]
        )

    def forward(self, x):
        return self._net(x)


class Decoder(nn.Module):
    """A feedforward encoder which upsamples its input."""

    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_channels,
            n_residual_blocks,
            residual_channels,
            stride,
    ):
        """Initializes a new Decoder instance.
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            hidden_channels: Number of channels in (non residual block) hidden layers.
            n_residual_blocks: Number of residual blocks in each residual stack.
            residual_channels: Number of hidden channels in residual blocks.
            stride: Stride to use in the upsampling (i.e. transpose) convolutions. Must
                be even.
        """
        super().__init__()

        assert stride % 2 == 0, '"stride" must be even.'

        net = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            ResidualStack(
                n_channels=hidden_channels,
                hidden_channels=residual_channels,
                n_residual_blocks=n_residual_blocks,
            ),
        ]
        for i in range(stride // 2):
            first, last = 0, stride // 2 - 1
            in_c = hidden_channels if i == first else hidden_channels // 2
            out_c = hidden_channels // 2 if i < last else out_channels
            net.append(
                nn.ConvTranspose2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            if i < last:
                net.append(nn.ReLU())
        self._net = nn.Sequential(*net)

    def forward(self, x):
        return self._net(x)


class VAEDecoder(nn.Module):
    def __init__(self,
                 out_channels=3,
                 latent_channels=512,
                 strides=[4],
                 hidden_channels=64,
                 residual_channels=32,
                 ):
        super(VAEDecoder, self).__init__()
        self._latent_channels = latent_channels
        decoder = []
        for i, stride in enumerate(reversed(strides)):
            in_c = self._latent_channels if i == 0 else hidden_channels
            out_c = hidden_channels if i < len(strides) - 1 else out_channels
            decoder.append(
                Decoder(
                    in_channels=in_c,
                    out_channels=out_c,
                    hidden_channels=hidden_channels,
                    residual_channels=residual_channels,
                    n_residual_blocks=2,
                    stride=stride,
                )
            )
        self._decoder = nn.Sequential(*decoder)

    def forward(self, x):
        y = self._decoder(x)
        y = (y.tanh() + 1.0) * 0.5
        return y


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

    md_type = 'dcgan'
    # md_type = 'vaedecoder'

    files = [f for f in os.listdir(folder_path) if re.search(r'.+\.pth$', f)]
    files = sorted(files)
    for fn in tqdm(files):
        if not fn.startswith('resnet'): continue

        pp = f'{name}_{fn}_dcgan.bin'
        if os.path.exists(pp): continue

        print(fn)

        model = load_model_from_path(fn, folder_path, dataset, kwargs)
        model.eval()

        if md_type == 'dcgan':
            netG = Generator(params)
        elif md_type == 'vaedecoder':
            netG = VAEDecoder(
                out_channels=params['nc'],
                latent_channels=params['nz'],
                residual_channels=params['ngf'],
                strides=[2, 2, 2, 2, 2],
            )
        else:
            raise 'Unknown model type'

        optimG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
        # optimG = optim.SGD(netG.parameters(), lr=1e-2)

        netG.cuda()
        netG.train()

        rand_beta = params['rand_beta']
        mixup = True
        randaug = True

        bar = tqdm(range(params['n_epochs']))
        for epoch in bar:
            loss_ema = None
            for data in dataset.loader['train']:
                x = dataset.get_data(data)[0]

                if mixup:
                    gamma = np.random.beta(rand_beta, rand_beta, x.size(0))
                    gamma_tensor = torch.from_numpy(gamma).float().cuda().reshape(-1, 1, 1, 1)
                    indices = torch.randperm(x.size(0), device='cuda', dtype=torch.long)
                    perm_x = x[indices]
                    nx = x * gamma_tensor + perm_x * (1 - gamma_tensor)
                else:
                    nx = x

                if randaug:
                    anchors = torch.rand(nx.shape, device='cuda')
                    inter = np.random.beta(rand_beta * 10, rand_beta, x.size(0))
                    inter_tensor = torch.from_numpy(gamma).float().cuda().reshape(-1, 1, 1, 1)
                    nx = nx * inter_tensor + anchors * (1 - inter_tensor)
                else:
                    nx = nx

                z = model.get_final_fm(nx)
                z_in = z.view(z.shape[0], z.shape[1], 1, 1)

                gx = netG(z_in.data)

                loss = torch.square(gx - nx.data).sum((1, 2, 3)).mean()

                optimG.zero_grad()
                loss.backward()
                optimG.step()

                lv = loss.item()
                if loss_ema is None:
                    loss_ema = lv
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * lv

                bar.set_description(f'epoch {epoch}: loss={lv:.3f}')

        pp = f'{name}_{fn}_{md_type}.bin'
        torch.save(netG, pp)
        del netG
        del model


def measure_wasserstein(dataset, folder_path):
    name = dataset.name

    img_folder = f'{name}_images'
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    tfunc = transforms.ToPILImage()

    files = [f for f in os.listdir(folder_path) if re.search(r'.+\.pth$', f)]
    files = sorted(files)
    for fn in tqdm(files):
        if not fn.startswith('resnet'): continue
        print(fn)

        w_model = GSW_NN(din=3 * 32 * 32, nofprojections=1, num_filters=32, model_depth=1)

        model = load_model_from_path(fn, folder_path, dataset, kwargs)
        model.eval()

        pp = os.path.join(f'{name}_tied_gm', f'tied_{fn}_gm.pkl')
        with open(pp, 'rb') as fh:
            gm = pickle.load(fh)

        pp = f'{name}_{fn}_dcgan.bin'
        netG = torch.load(pp)
        netG.eval()
        netG.cuda()

        with torch.no_grad():
            X, Y = list(), list()
            iters = 0
            for data in dataset.loader['train']:
                iters += 1
                x = dataset.get_data(data)[0]
                X.append(x)

                sX, sY = gm.sample(x.shape[0])
                sX_tensor = torch.from_numpy(sX).float().cuda()
                sX_tensor = torch.reshape(sX_tensor, list(sX_tensor.shape) + [1, 1])
                sI = netG(sX_tensor)

                Y.append(sI)
                print(iters)
                if iters > 2:
                    break

        X = torch.concat(X, dim=0)
        Y = torch.concat(Y, dim=0)
        X = X.reshape(X.shape[0], -1)
        Y = Y.reshape(Y.shape[0], -1)

        print(X.shape, Y.shape)

        Y = X + torch.randn(X.shape).float().cuda() * 1e-2
        dist = w_model.max_gsw(X.data, Y.data, iterations=10000)
        print(dist)

        exit(0)

        print(sI.shape)
        for k, si in enumerate(sI):
            pp = os.path.join(img_folder, f'{k}.png')
            im = tfunc(si)
            im.save(pp)
        exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    kwargs = parser.parse_args().__dict__

    env = trojanvision.environ.create(**kwargs)

    '''
    dataset = trojanvision.datasets.create(**kwargs)
    print('train DCGAN models for', dataset.name)
    name = dataset.name
    folder_path = f'./benign_{name}'
    train_DCGAN_models(dataset, folder_path)
    exit(0)
    # '''

    # '''
    dataset = trojanvision.datasets.create(**kwargs)
    print('measure DCGAN wasserstein for', dataset.name)
    name = dataset.name
    folder_path = f'./benign_{name}'
    measure_wasserstein(dataset, folder_path)
    # '''

    netG = VAEDecoder(
        out_channels=params['nc'],
        latent_channels=params['nz'],
        residual_channels=params['ngf'],
        strides=[2, 2, 2, 2, 2]
    )
    noise = torch.randn(1, params['nz'], 1, 1)
    o = netG(noise)
    print(o.shape)
    print(o.min(), o.max())
