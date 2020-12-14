# -*- coding: utf-8 -*-

# python backdoor_attack.py --attack badnet --verbose --pretrain --validate_interval 1 --epoch 50 --lr 1e-2 --mark_height 3 --mark_width 3 --mark_alpha 0.0

import trojanzoo.environ
import trojanzoo.dataset
import trojanzoo.model
import trojanzoo.trainer
import trojanzoo.mark
import trojanzoo.attack

from trojanzoo.environ import env
from trojanzoo.utils import summary
import argparse

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanzoo.environ.add_argument(parser)
    trojanzoo.dataset.add_argument(parser)
    trojanzoo.model.add_argument(parser)
    trojanzoo.trainer.add_argument(parser)
    trojanzoo.mark.add_argument(parser)
    trojanzoo.attack.add_argument(parser)
    args = parser.parse_args()

    trojanzoo.environ.create(**args.__dict__)
    dataset = trojanzoo.dataset.create(**args.__dict__)
    model = trojanzoo.model.create(dataset=dataset, **args.__dict__)
    trainer = trojanzoo.trainer.create(dataset=dataset, model=model, **args.__dict__)
    mark = trojanzoo.mark.create(dataset=dataset, **args.__dict__)
    attack = trojanzoo.attack.create(dataset=dataset, model=model, mark=mark, **args.__dict__)

    if env['verbose']:
        summary(dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack)
    attack.attack(**trainer)