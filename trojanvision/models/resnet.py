#!/usr/bin/env python3

from .imagemodel import _ImageModel, ImageModel
from trojanvision.utils.model_archs.resnet_s import ResNetS

import torch
import torch.nn as nn
from torch.utils import model_zoo
import torchvision.models
from torchvision.models.resnet import model_urls
from collections import OrderedDict
from collections.abc import Callable


class _ResNet(_ImageModel):

    def __init__(self, layer: int = 18, sub_type: str = '', **kwargs):
        super().__init__(**kwargs)
        layer = int(layer)
        module_list: list[nn.Module] = []
        if sub_type == 's':
            _model = ResNetS(nclasses=self.num_classes)
            module_list.append(('conv1', _model.conv1))
            module_list.append(('bn1', _model.bn1))
            module_list.append(('relu', nn.ReLU(inplace=True)))
            self.classifier = nn.Sequential(OrderedDict([
                ('fc', _model.linear)  # nn.Linear(512 * block.expansion, num_classes)
            ]))
        else:
            ModelClass: Callable[..., torchvision.models.ResNet] = getattr(torchvision.models, 'resnet' + str(layer))
            _model = ModelClass(num_classes=self.num_classes)
            if sub_type == 'comp':
                conv1: nn.Conv2d = _model.conv1
                _model.conv1 = nn.Conv2d(conv1.in_channels, conv1.out_channels,
                                         kernel_size=3, stride=1, padding=1, bias=False)
                module_list.append(('conv1', _model.conv1))
                module_list.append(('bn1', _model.bn1))
                module_list.append(('relu', _model.relu))
            else:
                module_list.append(('conv1', _model.conv1))
                module_list.append(('bn1', _model.bn1))
                module_list.append(('relu', _model.relu))
                module_list.append(('maxpool', _model.maxpool))
            self.pool = _model.avgpool  # nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(OrderedDict([
                ('fc', _model.fc)  # nn.Linear(512 * block.expansion, num_classes)
            ]))
            # block.expansion = 1 if BasicBlock and 4 if Bottleneck
            # ResNet 18,34 use BasicBlock, 50 and higher use Bottleneck
        module_list.extend([('layer1', _model.layer1),
                            ('layer2', _model.layer2),
                            ('layer3', _model.layer3),
                            ('layer4', _model.layer4)])
        self.features = nn.Sequential(OrderedDict(module_list))


class ResNet(ImageModel):

    def __init__(self, name: str = 'resnet', layer: int = 18,
                 model: type[_ResNet] = _ResNet, **kwargs):
        sub_type = ''
        if 'comp' in name:
            sub_type = 'comp'
        elif '_s' in name:
            sub_type = 's'
        super().__init__(name=name, layer=layer, model=model, sub_type=sub_type, **kwargs)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        url = model_urls[f'resnet{self.layer:d}']
        print('get official model weights from: ', url)
        _dict: OrderedDict[str, torch.Tensor] = model_zoo.load_url(url, **kwargs)
        new_dict = OrderedDict()
        for i, (key, value) in enumerate(_dict.items()):
            prefix = 'features.' if i < len(_dict) - 2 else 'classifier.'
            new_dict[prefix + key] = value
        return new_dict
