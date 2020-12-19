# -*- coding: utf-8 -*-

from .adv import *
from .poison import *
from .backdoor import *

from trojanvision.configs import Config, config
from trojanvision.datasets import ImageSet
import trojanzoo.attacks
from trojanzoo.attacks import Attack
import argparse
from typing import Union

class_dict = {
    # adversarial attack
    'pgd': PGD,
    # 'advmind': AdvMind,

    # poisoning attack
    'poison_basic': Poison_Basic,
    'imc_poison': IMC_Poison,

    # backdoor attack
    'badnet': BadNet,
    'trojannn': TrojanNN,
    'latent_backdoor': Latent_Backdoor,
    'imc': IMC,
    'reflection_backdoor': Reflection_Backdoor,
    'bypass_embed': Bypass_Embed,
    'trojannet': TrojanNet,
    'clean_label': Clean_Label,
    'hidden_trigger': Hidden_Trigger,

    'term_study': Term_Study,
    'unlearn': Unlearn,

    # imc adaptive settings
    'imc_latent': IMC_Latent,
    'imc_advtrain': IMC_AdvTrain,
    'imc_strip': IMC_STRIP,
    'imc_multi': IMC_Multi,
    'imc_magnet': IMC_MagNet,
    'imc_abs': IMC_ABS,
    'imc_adaptive': IMC_Adaptive,
}


def add_argument(parser: argparse.ArgumentParser, attack_name: str = None, attack: Union[str, Attack] = None,
                 class_dict: dict[str, type[Attack]] = class_dict):
    return trojanzoo.attacks.add_argument(parser=parser, attack_name=attack_name, attack=attack,
                                          class_dict=class_dict)


def create(attack_name: str = None, attack: Union[str, Attack] = None,
           dataset_name: str = None, dataset: Union[str, ImageSet] = None,
           config: Config = config, class_dict: dict[str, type[Attack]] = class_dict, **kwargs):
    return trojanzoo.attacks.create(attack_name=attack_name, attack=attack,
                                    dataset_name=dataset_name, dataset=dataset,
                                    config=config, class_dict=class_dict, **kwargs)