#!/usr/bin/env python3

from trojanzoo.attacks import Attack

from trojanvision.datasets.imageset import ImageSet
from trojanvision.models.imagemodel import ImageModel
from trojanvision.marks import Watermark
from trojanzoo.environ import env
from trojanzoo.utils.data import TensorListDataset, sample_batch
from trojanzoo.utils.logger import SmoothedValue

import torch
import torchvision.transforms.functional as F
import numpy as np
import functools
import math
import random
import os
import yaml

import argparse
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.utils.data


class BackdoorAttack(Attack):
    r"""Backdoor attack abstract class.
    It inherits :class:`trojanzoo.attacks.Attack`.

    Note:
        This class is actually equivalent to :class:`trojanvision.attacks.BadNet`.

    BackdoorAttack attaches a provided watermark to some training images
    and inject them into training set with target label.
    After retraining, the model will classify images with watermark
    of certain/all classes into target class.

    Args:
        mark (trojanvision.marks.Watermark): The watermark instance.
        target_class (int): The target class that images with watermark will be misclassified as.
            Defaults to ``0``.
        poison_percent (float): Percentage of poisoning inputs in the whole training set.
            Defaults to ``0.01``.
        train_mode (float): Training mode to inject backdoor.
            Choose from ``['batch', 'dataset', 'loss']``.
            Defaults to ``'batch'``.

            * ``'batch'``: For a clean batch, randomly picked :attr:`poison_num` inputs,
              attach trigger on them, modify their labels and append to original batch.
            * ``'dataset'``: Create a poisoned dataset and use the mixed dataset.
            * ``'loss'``: For a clean batch, calculate the loss on clean data
              and the loss on poisoned data (all batch)
              and mix them using :attr:`poison_percent` as weight.

    Attributes:
        poison_ratio (float): The ratio of poison data divided by clean data.
            ``poison_percent / (1 - poison_percent)``
        poison_num (float | int): The number of poison data in each batch / dataset.

            * ``train_mode == 'batch'  : poison_ratio * batch_size``
            * ``train_mode == 'dataset': int(poison_ratio * len(train_set))``
            * ``train_mode == 'loss'   : N/A``
        poison_set (torch.utils.data.Dataset):
            Poison dataset (no clean data) ``if train_mode == 'dataset'``.
    """

    name: str = 'backdoor_attack'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--target_class', type=int,
                           help='target class of backdoor '
                                '(default: 0)')
        group.add_argument('--source_class', type=int, nargs='+',
                           help='source class(es) of backdoor '
                                '(default: 1)')
        group.add_argument('--poison_percent', type=float,
                           help='malicious training data proportion '
                                '(default: 0.01)')
        group.add_argument('--train_mode', choices=['batch', 'dataset', 'loss'],
                           help='training mode to inject backdoor '
                                '(default: "batch")')
        return group

    def __init__(self, mark: Watermark = None,
                 source_class: list[int] = None,
                 target_class: int = 0, poison_percent: float = 0.01,
                 train_mode: str = 'batch', **kwargs):
        super().__init__(**kwargs)
        self.dataset: ImageSet
        self.model: ImageModel
        self.mark = mark
        self.param_list['backdoor'] = ['train_mode', 'target_class', 'source_class', 'poison_percent', 'poison_num']
        self.source_class = source_class
        self.target_class = target_class
        self.poison_percent = poison_percent
        self.poison_ratio = self.poison_percent / (1 - self.poison_percent)
        self.train_mode = train_mode
        match train_mode:
            case 'batch':
                self.poison_num = self.dataset.batch_size * self.poison_ratio
                self.poison_set = None
            case 'dataset':
                self.poison_num = int(len(self.dataset.loader['train'].dataset) * self.poison_ratio)
                self.poison_set = self.get_poison_dataset()
            case _:
                self.poison_set = None

    def attack(self, epochs: int, **kwargs):
        kwargs['validate_fn'] = kwargs.get('validate_fn', self.validate_fn)
        kwargs['save_fn'] = kwargs.get('save_fn', self.save)
        match self.train_mode:
            case 'batch':
                loader = self.dataset.get_dataloader(
                    'train', batch_size=self.dataset.batch_size + int(self.poison_num))
                return self.model._train(epochs, loader_train=loader,
                                         get_data_fn=self.get_data,
                                         **kwargs)
            case 'dataset':
                mix_dataset = torch.utils.data.ConcatDataset([self.dataset.loader['train'].dataset,
                                                              self.poison_set])
                loader = self.dataset.get_dataloader('train', dataset=mix_dataset)
                return self.model._train(epochs, loader_train=loader, **kwargs)
            case 'loss':
                if 'loss_fn' in kwargs.keys():
                    kwargs['loss_fn'] = functools.partial(self.loss_weighted, loss_fn=kwargs['loss_fn'])
                else:
                    kwargs['loss_fn'] = self.loss_weighted
                return self.model._train(epochs, **kwargs)
            case _:
                raise NotImplementedError(f'{self.train_mode=}')

    def get_poison_dataset(self, poison_label: bool = True,
                           poison_num: int = None,
                           seed: int = None
                           ) -> torch.utils.data.Dataset:
        r"""Get poison dataset (no clean data).

        Args:
            poison_label (bool):
                Whether to use target poison label for poison data.
                Defaults to ``True``.
            poison_num (int): Number of poison data.
                Defaults to ``round(self.poison_ratio * len(train_set))``
            seed (int): Random seed to sample poison input indices.
                Defaults to ``env['data_seed']``.

        Returns:
            torch.utils.data.Dataset:
                Poison dataset (no clean data).
        """
        if seed is None:
            seed = env['data_seed']
        torch.random.manual_seed(seed)
        train_set = self.dataset.loader['train'].dataset
        poison_num = poison_num or round(self.poison_ratio * len(train_set))
        _input, _label = sample_batch(train_set, batch_size=poison_num)
        _label = _label.tolist()

        if poison_label:
            _label = [self.target_class] * len(_label)
        trigger_input = self.add_mark(_input)
        return TensorListDataset(trigger_input, _label)

    def get_filename(self, mark_alpha: float = None, target_class: int = None, **kwargs) -> str:
        r"""Get filenames for current attack settings."""
        if mark_alpha is None:
            mark_alpha = self.mark.mark_alpha
        if target_class is None:
            target_class = self.target_class
        mark_filename = os.path.split(self.mark.mark_path)[-1]
        mark_name, mark_ext = os.path.splitext(mark_filename)
        _file = '{mark}_tar{target:d}_alpha{mark_alpha:.2f}_mark({mark_height:d},{mark_width:d})_pr{pr}'.format(
            mark=mark_name, target=target_class, mark_alpha=mark_alpha,
            mark_height=self.mark.mark_height, mark_width=self.mark.mark_width,
            pr=self.poison_percent
        )
        if self.mark.mark_random_pos:
            _file = 'randompos_' + _file
        if self.mark.mark_scattered:
            _file = 'scattered_' + _file
        return _file

    # ---------------------- I/O ----------------------------- #

    def save_params(self, file_path, obj=None):
        if obj is None:
            obj = self
        param_dict = dict()
        for key in obj.param_list:
            for pn in obj.param_list[key]:
                param_dict[pn] = getattr(obj, pn)
        with open(file_path, 'w') as fh:
            yaml.dump(param_dict, fh)
        return param_dict

    def load_params(self, file_path, obj=None):
        if not os.path.exists(file_path):
            return None
        if obj is None:
            obj = self
        with open(file_path, 'r') as fh:
            param_dict = yaml.load(fh, Loader=yaml.FullLoader)
        for pn in param_dict:
            setattr(obj, pn, param_dict[pn])
        return param_dict

    def save(self, filename: str = None, **kwargs):
        r"""Save attack results to files."""
        filename = filename or self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        np.save(file_path + '.npy', self.mark.mark.detach().cpu().numpy())
        F.to_pil_image(self.mark.mark).save(file_path + '.png')
        self.model.save(file_path + '.pth')
        self.save_params(file_path + '.yaml')
        self.save_params(file_path + '_mark.yaml', obj=self.mark)
        print('attack results saved at: ', file_path)

    def load(self, filename: str = None, **kwargs):
        r"""Load attack results from previously saved files."""
        filename = filename or self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        self.mark.load_mark(file_path + '.npy', already_processed=True)
        self.model.load(file_path + '.pth')
        self.load_params(file_path + '.yaml')
        self.load_params(file_path + '_mark.yaml', obj=self.mark)
        print('attack results loaded from: ', file_path)

    # ---------------------- Utils ---------------------------- #

    def add_mark(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Add watermark to input tensor.
        Defaults to :meth:`trojanvision.marks.Watermark.add_mark()`.
        """
        return self.mark.add_mark(x, **kwargs)

    def loss_weighted(self, _input: torch.Tensor = None, _label: torch.Tensor = None,
                      _output: torch.Tensor = None, loss_fn: Callable[..., torch.Tensor] = None,
                      **kwargs) -> torch.Tensor:
        loss_fn = loss_fn if loss_fn is not None else self.model.loss
        loss_clean = loss_fn(_input, _label, **kwargs)
        trigger_input = self.add_mark(_input)
        trigger_label = self.target_class * torch.ones_like(_label)
        loss_poison = loss_fn(trigger_input, trigger_label, **kwargs)
        return (1 - self.poison_percent) * loss_clean + self.poison_percent * loss_poison

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor],
                 org: bool = False, keep_org: bool = True,
                 poison_label: bool = True, **kwargs
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Get data.

        Args:
            data (tuple[torch.Tensor, torch.Tensor]): Tuple of input and label tensors.
            org (bool): Whether to return original clean data directly.
                Defaults to ``False``.
            keep_org (bool): Whether to keep original clean data in final results.
                If ``False``, the results are all infected.
                Defaults to ``True``.
            poison_label (bool): Whether to use target class label for poison data.
                Defaults to ``True``.
            **kwargs: Any keyword argument (unused).

        Returns:
            (torch.Tensor, torch.Tensor): Result tuple of input and label tensors.
        """
        return self.get_data_from_source_classes(data, org, keep_org, poison_label, **kwargs)

        _input, _label = self.model.get_data(data)
        if not org:
            if keep_org:
                decimal, integer = math.modf(len(_label) * self.poison_ratio)
                integer = int(integer)
                if random.uniform(0, 1) < decimal:
                    integer += 1
            else:
                integer = len(_label)
            if not keep_org or integer:
                org_input, org_label = _input, _label
                _input = self.add_mark(org_input[:integer])
                _label = org_label[:integer]
                if poison_label:
                    _label = self.target_class * torch.ones_like(_label)
                if keep_org:
                    _input = torch.cat((_input, org_input))
                    _label = torch.cat((_label, org_label))
        return _input, _label

    def validate_fn(self,
                    get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
                    loss_fn: Callable[..., torch.Tensor] = None,
                    main_tag: str = 'valid', indent: int = 0,
                    threshold: float = 5.0,
                    **kwargs) -> tuple[float, float]:
        clean_acc, _ = self.model._validate(print_prefix='Validate Clean', main_tag='valid clean',
                                            get_data_fn=None, indent=indent, **kwargs)
        asr, _ = self.model._validate(print_prefix='Validate ASR', main_tag='valid asr',
                                      get_data_fn=self.get_data, keep_org=False, poison_label=True,
                                      indent=indent, **kwargs)
        # self.model._validate(print_prefix='Validate Trigger Org', main_tag='',
        #                      get_data_fn=self.get_data, keep_org=False, poison_label=False,
        #                      indent=indent, **kwargs)
        # prints(f'Validate Confidence: {self.validate_confidence():.3f}', indent=indent)
        # prints(f'Neuron Jaccard Idx: {self.get_neuron_jaccard():.3f}', indent=indent)
        if self.clean_acc - clean_acc > threshold:
            asr = 0.0
        return asr, clean_acc

    @torch.no_grad()
    def validate_confidence(self, mode: str = 'valid', success_only: bool = True) -> float:
        r"""Get :attr:`self.target_class` confidence on dataset of :attr:`mode`.

        Args:
            mode (str): Dataset mode. Defaults to ``'valid'``.
            success_only (bool): Whether to only measure confidence
                on attack-successful inputs.
                Defaults to ``True``.

        Returns:
            float: Average confidence of :attr:`self.target_class`.
        """
        source_class = self.source_class or list(range(self.dataset.num_classes))
        source_class = source_class.copy()
        if self.target_class in source_class:
            source_class.remove(self.target_class)
        loader = self.dataset.get_dataloader(mode=mode, class_list=source_class)

        confidence = SmoothedValue()
        for data in loader:
            _input, _label = self.model.get_data(data)
            trigger_input = self.add_mark(_input)
            trigger_label = self.model.get_class(trigger_input)
            if success_only:
                trigger_input = trigger_input[trigger_label == self.target_class]
                if len(trigger_input) == 0:
                    continue
            batch_conf = self.model.get_prob(trigger_input)[:, self.target_class].mean()
            confidence.update(batch_conf, len(trigger_input))
        return confidence.global_avg

    @torch.no_grad()
    def get_neuron_jaccard(self, k: int = None, ratio: float = 0.5) -> float:
        r"""Get Jaccard Index of neuron activations for feature maps
        between normal inputs and poison inputs.

        Find average top-k neuron indices of 2 kinds of feature maps
        ``clean_idx and poison_idx``, and return
        :math:`\frac{\text{len(clean\_idx \& poison\_idx)}}{\text{len(clean\_idx | poison\_idx)}}`

        Args:
            k (int): Top-k neurons to calculate jaccard index.
                Defaults to ``None``.
            ratio (float): Percentage of neurons if :attr:`k` is not provided.
                Defaults to ``0.5``.

        Returns:
            float: Jaccard Index.
        """
        clean_feats_list = []
        poison_feats_list = []
        for data in self.dataset.loader['valid']:
            _input, _label = self.model.get_data(data)
            trigger_input = self.add_mark(_input)

            clean_feats = self.model.get_fm(_input)
            poison_feats = self.model.get_fm(trigger_input)
            if clean_feats.dim() > 2:
                clean_feats = clean_feats.flatten(2).mean(2)
                poison_feats = poison_feats.flatten(2).mean(2)
            clean_feats_list.append(clean_feats)
            poison_feats_list.append(poison_feats)
        clean_feats_list = torch.cat(clean_feats_list).mean(dim=0)
        poison_feats_list = torch.cat(poison_feats_list).mean(dim=0)

        k = k or int(len(clean_feats_list) * ratio)
        clean_idx = set(clean_feats_list.argsort(
            descending=True)[:k].detach().cpu().tolist())
        poison_idx = set(poison_feats_list.argsort(
            descending=True)[:k].detach().cpu().tolist())
        jaccard_idx = len(clean_idx & poison_idx) / len(clean_idx | poison_idx)
        return jaccard_idx

    def get_source_class(self):
        if self.source_class is None:
            source_class = self.source_class or list(range(self.dataset.num_classes))
            if self.target_class in source_class:
                source_class.remove(self.target_class)
            self.source_class = source_class
        return self.source_class

    def get_source_class_dataset(self, mode='train') -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        source_class = self.get_source_class()
        dataset = self.dataset.get_dataset(mode, class_list=source_class)
        return dataset

    def get_source_inputs_index(self, _label, source_class=None):
        if source_class is None:
            source_class = self.get_source_class()
        idx = None
        for c in source_class:
            _idx = _label.eq(c)
            if idx is None:
                idx = _idx
            else:
                idx = torch.logical_or(idx, _idx)
        return idx

    def get_data_from_source_classes(self, data: tuple[torch.Tensor, torch.Tensor],
                                     org: bool = False, keep_org: bool = True,
                                     poison_label: bool = True, **kwargs
                                     ) -> tuple[torch.Tensor, torch.Tensor]:

        _input, _label = self.model.get_data(data)
        if not org:
            src_idx = self.get_source_inputs_index(_label).cpu().detach().numpy()
            if np.sum(src_idx) <= 0:
                return _input, _label
            src_idx = np.arange(len(_label))[src_idx]
            if keep_org:
                decimal, integer = math.modf(len(src_idx) * self.poison_ratio)
                integer = int(integer)
                if random.uniform(0, 1) < decimal:
                    integer += 1
            else:
                integer = len(src_idx)
            if not keep_org or integer:
                org_input, org_label = _input, _label
                _input = self.add_mark(org_input[src_idx[:integer]])
                _label = org_label[src_idx[:integer]]
                if poison_label:
                    _label = self.target_class * torch.ones_like(_label)
                if keep_org:
                    _input = torch.cat((_input, org_input))
                    _label = torch.cat((_label, org_label))
        return _input, _label

    def get_poison_dataset_from_source_classes(self, poison_label: bool = True,
                                               poison_num: int = None,
                                               seed: int = None,
                                               mode: str = 'train',
                                               ) -> torch.utils.data.Dataset:
        if seed is None:
            seed = env['data_seed']
        torch.random.manual_seed(seed)
        dataset = self.get_source_class_dataset(mode=mode)
        poison_num = poison_num or round(self.poison_ratio * len(dataset))
        dataset, _ = self.dataset.split_dataset(dataset, length=poison_num)
        loader = self.dataset.get_dataloader(mode, dataset=dataset)

        def trans_fn(data):
            _input, _label = self.model.get_data(data)
            _input = self.add_mark(_input)
            return _input, _label

        _input_tensor, _label_list = self.expand_loader_to_tensor_and_list(loader, trans_fn=trans_fn)

        if poison_label:
            _label_list = [self.target_class] * len(_label_list)
        return TensorListDataset(_input_tensor, _label_list)

    def expand_loader_to_tensor_and_list(self, loader, trans_fn=None):
        _label_list = list()
        _input_list = list()
        if trans_fn is None:
            trans_fn = self.model.get_data
        for data in loader:
            _input, _label = trans_fn(data)
            _input_list.append(_input.detach().cpu())
            _label_list.append(_label.detach().cpu())
        _input_tensor = torch.cat(_input_list)
        _label_list = torch.cat(_label_list)
        _label_list = _label_list.tolist()
        return _input_tensor, _label_list


class CleanLabelBackdoor(BackdoorAttack):
    r"""Backdoor attack abstract class of clean label.
    It inherits :class:`trojanvision.attacks.BackdoorAttack`.

    Under clean-label setting, only clean inputs from target class are infected,
    while the distortion is negligible for human to detect.
    """
    name = 'clean_label'

    def __init__(self, *args, train_mode: str = 'dataset', **kwargs):
        # monkey patch: to avoid calling get_poison_dataset() in super().__init__
        train_mode = 'batch'
        super().__init__(*args, train_mode=train_mode, **kwargs)
        self.target_set = self.dataset.get_dataset('train', class_list=[self.target_class])
        self.poison_num = int(self.poison_ratio * len(self.target_set))
        self.train_mode = 'dataset'

    def get_poison_dataset(self, poison_num: int = None, load_mark: bool = True,
                           seed: int = None) -> torch.utils.data.Dataset:
        r"""Get poison dataset from target class (no clean data).

        Args:
            poison_num (int): Number of poison data.
                Defaults to ``self.poison_num``
            load_mark (bool): Whether to load previously saved watermark.
                This should be ``False`` during attack.
                Defaults to ``True``.
            seed (int): Random seed to sample poison input indices.
                Defaults to ``env['data_seed']``.

        Returns:
            torch.utils.data.Dataset:
                Poison dataset from target class (no clean data).
        """
        file_path = os.path.join(self.folder_path, self.get_filename() + '.npy')
        if load_mark:
            if os.path.isfile(file_path):
                self.load_mark = False
                self.mark.load_mark(file_path, already_processed=True)
            else:
                raise FileNotFoundError(file_path)
        if seed is None:
            seed = env['data_seed']
        torch.random.manual_seed(seed)
        poison_num = min(poison_num or self.poison_num, len(self.target_set))
        _input, _label = sample_batch(self.target_set, batch_size=poison_num)
        _label = _label.tolist()
        trigger_input = self.add_mark(_input)
        return TensorListDataset(trigger_input, _label)


class DynamicBackdoor(BackdoorAttack):
    name = 'dynamic_backdoor'
