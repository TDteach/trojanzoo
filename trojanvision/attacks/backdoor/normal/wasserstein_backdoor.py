import torch.nn as nn
import torchvision
import os

from ...abstract import BackdoorAttack

from trojanzoo.utils.logger import MetricLogger
from trojanzoo.utils.output import ansi, get_ansi_len, output_iter
from trojanzoo.utils.data import TensorListDataset, sample_batch
from trojanzoo.environ import env
import trojanvision

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import math
import random
import numpy as np
import copy

from typing import TYPE_CHECKING
import argparse
from collections.abc import Callable

from tqdm import tqdm

if TYPE_CHECKING:
    import torch.utils.data

loss_fn = nn.CrossEntropyLoss()


class ArgClass():
    def __init__(self, kwargs):
        self.__dict__.update(kwargs)


class WasserteinBackdoor(BackdoorAttack):
    name: str = 'wasserstein_backdoor'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--step_one_iterations', type=int,
                           help='iterations of step one where poison_epoch and trigger_epoch alternatively run'
                                '(default: 50)')
        group.add_argument('--step_two_iterations', type=int,
                           help='iterations of step two where only poison_epoch runs'
                                '(default: 10)')
        group.add_argument('--pgd_eps', type=int,
                           help='|noise|_{\infinity} <= pgd_esp '
                                '(default: 0.1)')
        return group

    def __init__(self,
                 step_one_iterations: int = 50,
                 step_two_iterations: int = 10,
                 pgd_eps: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)

        self.param_list['wasserstein_backdoor'] = ['step_one_iterations', 'step_two_iterations', 'pgd_eps']
        self.step_one_iterations = step_one_iterations
        self.step_two_iterations = step_two_iterations
        self.pgd_eps = pgd_eps

        self.device = env['device']

        # data_channel = self.dataset.data_shape[0]
        # image_size = self.dataset.data_shape[1]
        # self.trigger_generator = self.get_trigger_generator(in_channels=data_channel, image_size=image_size)
        self.atkmodel = UNet(3).to(self.device)
        self.tgtmodel = UNet(3).to(self.device)
        self.tgtoptimizer = optim.Adam(self.tgtmodel.parameters(), lr=5e-4)

        assert len(self.model._model.classifier) == 1

        source_class = self.source_class or list(range(self.dataset.num_classes))
        source_class = source_class.copy()
        if self.target_class in source_class:
            source_class.remove(self.target_class)
        self.source_class = source_class

    def step_one(self, **kwargs):
        train_loader = self.dataset.get_dataloader(mode='train')
        # train_loader = self.dataset.get_dataloader(mode='valid')
        test_loader = self.dataset.get_dataloader(mode='valid')

        basepath = self.folder_path
        # if not os.path.exists(basepath):
        #     print(f'Creating new model training in {basepath}')
        #     os.makedirs(basepath)
        checkpoint_path = os.path.join(basepath, 'checkpoint.ckpt')
        bestmodel_path = os.path.join(basepath, 'bestmodel.ckpt')

        params = {
            'device': 'cuda',
            'batch_size': 128,
            'epochs': self.step_one_iterations,
            'lr': 0.01,
            'lr_atk': 0.0005,
            'train_epoch': 1,
            'target_label': 0,
            'eps': self.pgd_eps,
            'alpha': 0.5,
            'attack_portion': 1.0,
            'epochs_per_external_eval': 10,
            'cls_test_epochs': 5,
            'best_threshold': 0.1,
            'avoid_cls_reinit': True,
            'test_eps': None,
            'test_alpha': None,
        }
        args = ArgClass(params)
        post_transforms = None

        best_acc_clean = 0
        best_acc_poison = 0
        trainlosses = []
        start_epoch = 1

        atkmodel = self.atkmodel
        tgtmodel = self.tgtmodel
        tgtoptimizer = self.tgtoptimizer
        clsmodel = self.model._model
        model_name = self.model.name

        if True:
            writer = None
            target_transform = get_target_transform(self.target_class)
            clip_image = get_clip_image()
            clsoptimizer = kwargs['optimizer']
            for epoch in range(start_epoch, args.epochs + 1):
                for i in range(args.train_epoch):
                    print(f'===== EPOCH: {epoch}/{args.epochs + 1} CLS {i + 1}/{args.train_epoch} =====')
                    if not args.avoid_cls_reinit:
                        clsoptimizer = optim.SGD(clsmodel.parameters(), lr=args.lr)
                    trainloss = train(args, atkmodel, tgtmodel, clsmodel, tgtoptimizer, clsoptimizer, target_transform,
                                      train_loader,
                                      epoch, i, writer, clip_image,
                                      post_transforms=post_transforms)
                    trainlosses.append(trainloss)
                atkmodel.load_state_dict(tgtmodel.state_dict())
                if not args.avoid_cls_reinit:
                    # clsmodel = create_net().to(args.device)
                    # scratchmodel = create_net().to(args.device)
                    pass
                else:
                    zz = clsmodel.__class__
                    scratchmodel = zz(name=model_name, dataset=self.dataset, num_classes=clsmodel.num_classes).to(
                        args.device)
                    scratchmodel.load_state_dict(clsmodel.state_dict())  # transfer from cls to scratch for testing

                if epoch % args.epochs_per_external_eval == 0 or epoch == args.epochs:
                    acc_clean, acc_poison = test(args, atkmodel, scratchmodel, target_transform,
                                                 train_loader, test_loader, epoch, args.cls_test_epochs, writer,
                                                 clip_image,
                                                 log_prefix='External')
                else:
                    acc_clean, acc_poison = test(args, atkmodel, scratchmodel, target_transform,
                                                 train_loader, test_loader, epoch, args.train_epoch, writer, clip_image,
                                                 log_prefix='Internal')

                if acc_clean > best_acc_clean or (
                        acc_clean + args.best_threshold > best_acc_clean and best_acc_poison < acc_poison):
                    best_acc_poison = acc_poison
                    best_acc_clean = acc_clean
                    torch.save({'atkmodel': atkmodel.state_dict(), 'clsmodel': clsmodel.state_dict()}, bestmodel_path)

                torch.save({
                    'atkmodel': atkmodel.state_dict(),
                    'clsmodel': clsmodel.state_dict(),
                    'tgtoptimizer': tgtoptimizer.state_dict(),
                    'best_acc_clean': best_acc_clean,
                    'best_acc_poison': best_acc_poison,
                    'trainlosses': trainlosses,
                    'epoch': epoch
                }, checkpoint_path)

    def attack(self, **kwargs):
        # other_set = self.get_source_class_dataset()
        # other_loader = self.dataset.get_dataloader(mode='train', dataset=other_set)

        print('Step one')
        self.step_one(**kwargs)
        # kwargs['epochs'] = self.train_poison_epochs
        # for _iter in range(self.step_one_iterations):
        #     self.train_poison_model(**kwargs)
        #     self.train_trigger_generator(other_loader)
        print('Step two')
        kwargs['epochs'] = self.train_poison_epochs * self.step_two_iterations
        ret = self.train_poison_model(**kwargs)
        return ret

    def train_poison_model(self, **kwargs):
        self.atkmodel.eval()
        self.model.train()
        ret = super().attack(**kwargs)
        return ret

    def get_trigger_noise(self, _input: torch.Tensor) -> torch.Tensor:
        noise = self.atkmodel(_input)
        return noise
        # raw_output = self.trigger_generator(_input)
        # trigger_output = raw_output.tanh() / 2.0 + 0.5
        # return trigger_output - _input

    def train_trigger_generator(self, other_loader, verbose: bool = True):

        normalized_weight = self.model._model.classifier[0].weight
        normalized_weight = torch.transpose(normalized_weight, 0, 1)
        normalized_weight = torch.nn.functional.normalize(normalized_weight, dim=0).data

        r"""Train :attr:`self.trigger_generator`."""
        # optimizer = torch.optim.Adam(self.trigger_generator.parameters(), lr=1e-2, betas=(0.5, 0.9))
        optimizer = torch.optim.SGD(self.trigger_generator.parameters(), lr=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.train_trigger_epochs)
        loader = other_loader
        logger = MetricLogger()
        logger.create_meters(loss=None, ce=None, w2d=None)
        print_prefix = 'Trigger Epoch'

        self.model.eval()
        self.trigger_generator.train()
        for _epoch in range(self.train_trigger_epochs):

            _epoch += 1
            logger.reset()
            header: str = '{blue_light}{0}: {1}{reset}'.format(
                print_prefix, output_iter(_epoch, self.train_trigger_epochs), **ansi)
            header = header.ljust(max(len(print_prefix), 30) + get_ansi_len(header))

            self.trigger_generator.train()
            for data in logger.log_every(loader, header=header) if verbose else loader:
                optimizer.zero_grad()
                _input, _label = self.model.get_data(data)
                batch_size = len(_input)

                _trigger_input = self.add_mark(_input)

                _double_input = torch.cat([_input, _trigger_input], axis=0)
                _double_fm = self.model.get_layer(_double_input, layer_output="flatten")

                # -------------Dxx sliced-wasserstein distance (DSWD)----------------
                _double_proj = torch.matmul(_double_fm, normalized_weight)
                _benign_proj = _double_proj[:batch_size]
                _trigger_proj = _double_proj[batch_size:]

                x1, _ = torch.sort(_benign_proj, dim=0)
                y1, _ = torch.sort(_trigger_proj, dim=0)
                z = x1 - y1
                w2d_vec = torch.mean(torch.square(z), dim=0)
                loss_w2d = torch.mean(w2d_vec)

                _double_logits = self.model.get_layer(_double_fm, layer_input="flatten", layer_output="output")

                _trigger_label = self.target_class * torch.ones_like(_label)
                _double_label = torch.cat([_label, _trigger_label], axis=0)

                loss_ce = torch.nn.functional.cross_entropy(_double_logits, _double_label)

                loss = loss_ce + loss_w2d
                loss.backward()
                optimizer.step()
                logger.update(n=batch_size, loss=loss.item(), ce=loss_ce.item(), w2d=loss_w2d.item())
            lr_scheduler.step()
            self.trigger_generator.eval()
        optimizer.zero_grad()

    def get_source_class_dataset(self, mode='train') -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        source_class = self.source_class or list(range(self.dataset.num_classes))
        source_class = source_class.copy()
        if self.target_class in source_class:
            source_class.remove(self.target_class)
        dataset = self.dataset.get_dataset(mode, class_list=source_class)
        return dataset

    def get_source_inputs_index(self, _label):
        idx = None
        for c in self.source_class:
            _idx = _label.eq(c)
            if idx is None:
                idx = _idx
            else:
                idx = torch.logical_or(idx, _idx)
        return idx

    # -------------------------------- I/O ------------------------------ #

    def save(self, filename: str = None, **kwargs):
        r"""Save attack results to files."""
        filename = filename or self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        torch.save(self.atkmodel.state_dict(), file_path + '_trigger.pth')
        self.model.save(file_path + '.pth')
        self.save_params(file_path + '.yaml')
        print('attack results saved at: ', file_path)

    def load(self, filename: str = None, **kwargs):
        r"""Load attack results from previously saved files."""
        filename = filename or self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        self.atkmodel.load_state_dict(torch.load(file_path + '_trigger.pth'))
        self.model.load(file_path + '.pth')
        self.load_params(file_path + '.yaml')
        print('attack results loaded from: ', file_path)

    # -------------------------------- Trigger Generator ------------------------------ #

    @staticmethod
    def get_trigger_generator(in_channels: int = 3, image_size: int = 32) -> torch.nn.Module:
        a = None
        match image_size:
            case 28:
                a = [64, 128]
            case 32:
                a = [64, 128]
            case 224:
                a = [64, 128, 256, 512, 1024]
            case _:
                raise NotImplementedError

        a.reverse()
        dec_chs = tuple(a)
        a.append(in_channels)
        a.reverse()
        enc_chs = tuple(a)

        model = UNet(enc_chs=enc_chs, dec_chs=dec_chs, input_channel=in_channels).to(device=env['device']).eval()
        return model

    # -------------------------------- override functions ------------------------------ #
    def add_mark(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        _noise = self.get_trigger_noise(x)
        _trigger_x = x + self.pgd_eps * _noise
        return torch.clip(_trigger_x, min=0.0, max=1.0)

    def validate_fn(self,
                    get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
                    loss_fn: Callable[..., torch.Tensor] = None,
                    main_tag: str = 'valid', indent: int = 0,
                    threshold: float = 3.0,
                    **kwargs) -> tuple[float, float]:
        clean_acc, _ = self.model._validate(print_prefix='Validate Clean', main_tag='valid clean',
                                            get_data_fn=None, indent=indent, **kwargs)
        asr, _ = self.model._validate(print_prefix='Validate ASR', main_tag='valid asr',
                                      get_data_fn=self.get_data, keep_org=False, poison_label=True,
                                      indent=indent, **kwargs)
        return clean_acc + asr, clean_acc

    def get_filename(self, mark_alpha: float = None, target_class: int = None, **kwargs) -> str:
        r"""Get filenames for current attack settings."""
        target_class = self.target_class
        source_class = self.source_class
        _file = 'wb_tar{target:d}_src{source}pgd{pgd_eps:.2f}_pr{pr}'.format(
            target=target_class, source=source_class, pgd_eps=self.pgd_eps, pr=self.poison_percent)
        return _file

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor],
                 org: bool = False, keep_org: bool = True,
                 poison_label: bool = True, **kwargs
                 ) -> tuple[torch.Tensor, torch.Tensor]:

        return super().get_data_from_source_classes(data, org, keep_org, poison_label, **kwargs)

    def loss_weighted(self, _input: torch.Tensor = None, _label: torch.Tensor = None,
                      _output: torch.Tensor = None, loss_fn: Callable[..., torch.Tensor] = None,
                      **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def get_poison_dataset(self, poison_label: bool = True,
                           poison_num: int = None,
                           seed: int = None
                           ) -> torch.utils.data.Dataset:

        return super().get_poison_dataset_from_source_classes(poison_label, poison_num, seed)


def DSWD_dis(clean_feat, poi_feat, weight):
    # clean_len = len(clean_feat)
    # poi_len = len(poi_feat)
    clean_feat = clean_feat.transpose(0, 1)
    poi_feat = poi_feat.transpose(0, 1)
    proj_clean_feat = weight.mm(clean_feat)
    proj_poi_feat = weight.mm(poi_feat)
    class_num = proj_clean_feat.size(0)
    dis = []
    for i in range(class_num):
        proj_clean_tmp, _ = torch.sort(proj_clean_feat[i, :])
        proj_poi_tmp, _ = torch.sort(proj_poi_feat[i, :])
        d = torch.abs(proj_clean_tmp - proj_poi_tmp)
        dis.append(torch.mean(d))
    dswd = torch.mean(torch.stack(dis))

    return dswd


def get_clip_image():
    def _clip_image(x):
        return torch.clamp(x, 0, 1.0)

    return _clip_image


def get_target_transform(_attack_target=1):
    def _all2one(x, attack_target=_attack_target):
        return torch.ones_like(x) * attack_target

    return _all2one


def train(args, atkmodel, tgtmodel, clsmodel, tgtoptimizer, clsoptimizer, target_transform,
          train_loader, epoch, train_epoch, writer, clip_image, post_transforms=None):
    atkmodel.eval()
    clsmodel.train()
    tgtmodel.train()

    losslist = []
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
    for batch_idx, (data, target) in pbar:
        bs = data.size(0)
        if post_transforms is not None:
            data = post_transforms(data)

        ########################################
        #### Update Trigger Function ####
        ########################################

        data, target = data.to(args.device), target.to(args.device)
        noise = tgtmodel(data) * args.eps
        atkdata = clip_image(data + noise)
        atktarget = target_transform(target)
        if args.attack_portion < 1.0:
            atkdata = atkdata[:int(args.attack_portion * bs)]
            atktarget = atktarget[:int(args.attack_portion * bs)]

        # Calculate loss
        output, clean_feat = clsmodel(data, return_final_fm=True)
        atkoutput, poison_feat = clsmodel(atkdata, return_final_fm=True)
        sd = clsmodel.state_dict()
        sd_keys = clsmodel.state_dict().keys()
        weight_tensor = clsmodel.classifier.state_dict()['fc.weight']
        # print(weight_tensor.keys())
        loss_poison = loss_fn(atkoutput, atktarget)
        loss_DSWD = DSWD_dis(clean_feat, poison_feat, weight_tensor)
        loss1 = loss_poison + loss_DSWD
        losslist.append(loss_poison.item())

        clsoptimizer.zero_grad()
        tgtoptimizer.zero_grad()
        loss1.backward(retain_graph=True)
        tgtoptimizer.step()  # this is the slowest step

        ###############################
        #### Update the classifier ####
        ###############################
        noise = atkmodel(data) * args.eps
        atkdata = clip_image(data + noise)
        atktarget = target_transform(target)
        if args.attack_portion < 1.0:
            atkdata = atkdata[:int(args.attack_portion * bs)]
            atktarget = atktarget[:int(args.attack_portion * bs)]

        atkoutput = clsmodel(atkdata)
        loss_clean = loss_fn(output, target)
        loss_poison = loss_fn(atkoutput, atktarget)
        loss2 = loss_clean * args.alpha + (1 - args.alpha) * loss_poison
        clsoptimizer.zero_grad()
        loss2.backward()
        clsoptimizer.step()

        if batch_idx % 10 == 0 or batch_idx == (len(train_loader) - 1):
            pbar.set_description('Train [{}] Loss: clean {:.4f} poison {:.4f} CLS {:.4f} ATK:{:.4f}'.format(
                epoch, loss_clean.item(), loss_poison.item(), loss1.item(), loss2.item()))
    pbar.close()
    atkloss = sum(losslist) / len(losslist)
    if writer is not None:
        writer.add_scalar('train/loss(atk)', atkloss,
                          global_step=(epoch - 1) * args.train_epoch + train_epoch)

    batch_img = torch.cat(
        [data[:16].clone().cpu(), noise[:16].clone().cpu(), atkdata[:16].clone().cpu()], 0)
    batch_img = F.upsample(batch_img, scale_factor=(4, 4))
    grid = torchvision.utils.make_grid(batch_img, normalize=True)
    if writer is not None:
        writer.add_image("Train Images", grid, global_step=(epoch - 1) * args.train_epoch + train_epoch)

    return atkloss


def test(args, atkmodel, scratchmodel, target_transform,
         train_loader, test_loader, epoch, trainepoch, writer, clip_image,
         testoptimizer=None, log_prefix='Internal', epochs_per_test=5):
    # default phase 2 parameters to phase 1
    if args.test_alpha is None:
        args.test_alpha = args.alpha
    if args.test_eps is None:
        args.test_eps = args.eps

    atkmodel.eval()
    scratchmodel.train()

    if testoptimizer is None:
        testoptimizer = optim.SGD(scratchmodel.parameters(), lr=args.lr)
    for cepoch in range(trainepoch):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
        for batch_idx, (data, target) in pbar:
            bs = data.size(0)
            data, target = data.to(args.device), target.to(args.device)
            testoptimizer.zero_grad()
            with torch.no_grad():
                noise = atkmodel(data) * args.test_eps
                atkdata = clip_image(data + noise)
                atktarget = target_transform(target)
                if args.attack_portion < 1.0:
                    atkdata = atkdata[:int(args.attack_portion * bs)]
                    atktarget = atktarget[:int(args.attack_portion * bs)]

            atkoutput = scratchmodel(atkdata)
            output = scratchmodel(data)

            loss_clean = loss_fn(output, target)
            loss_poison = loss_fn(atkoutput, atktarget)

            loss = args.alpha * loss_clean + (1 - args.test_alpha) * loss_poison

            loss.backward()
            testoptimizer.step()

            if batch_idx % 10 == 0 or batch_idx == (len(train_loader) - 1):
                pbar.set_description(
                    'Test [{}-{}] Loss: Clean {:.4f} Poison {:.4f} Total {:.5f}'.format(
                        epoch, cepoch,
                        loss_clean.item(),
                        loss_poison.item(),
                        loss.item()
                    ))
        if cepoch % epochs_per_test == 0 or cepoch == trainepoch - 1:
            correct = 0
            correct_transform = 0
            test_loss = 0
            test_transform_loss = 0

            with torch.no_grad():
                for data, target in test_loader:
                    bs = data.size(0)
                    data, target = data.to(args.device), target.to(args.device)
                    output = scratchmodel(data)
                    test_loss += loss_fn(output, target).item() * bs  # sum up batch loss
                    pred = output.max(1, keepdim=True)[
                        1]  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

                    noise = atkmodel(data) * args.test_eps
                    atkdata = clip_image(data + noise)
                    atkoutput = scratchmodel(atkdata)
                    test_transform_loss += loss_fn(atkoutput, target_transform(target)).item() * bs  # sum up batch loss
                    atkpred = atkoutput.max(1, keepdim=True)[
                        1]  # get the index of the max log-probability
                    correct_transform += atkpred.eq(
                        target_transform(target).view_as(atkpred)).sum().item()

            test_loss /= len(test_loader.dataset)
            test_transform_loss /= len(test_loader.dataset)

            correct /= len(test_loader.dataset)
            correct_transform /= len(test_loader.dataset)

            print(
                '\n{}-Test set [{}]: Loss: clean {:.4f} poison {:.4f}, Accuracy: clean {:.2f} poison {:.2f}'.format(
                    log_prefix, cepoch,
                    test_loss, test_transform_loss,
                    correct, correct_transform
                ))

    if writer is not None:
        writer.add_scalar(f'{log_prefix}-acc(clean)', correct,
                          global_step=epoch - 1)
        writer.add_scalar(f'{log_prefix}-acc(poison)',
                          correct_transform,
                          global_step=epoch - 1)

        batch_img = torch.cat(
            [data[:16].clone().cpu(), noise[:16].clone().cpu(), atkdata[:16].clone().cpu()], 0)
        batch_img = F.upsample(batch_img, scale_factor=(4, 4))
        grid = torchvision.utils.make_grid(batch_img, normalize=True)
        writer.add_image(f"{log_prefix}-Test Images", grid, global_step=(epoch - 1))

    return correct, correct_transform


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding='same')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding='same')

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, out_channel):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, out_channel, 1),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        out = torch.tanh(out)

        return out
