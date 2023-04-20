import trojanvision
import argparse
import os
import re
import time
import math

import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def update_models_info(models_info, name, num=None):
    if name not in models_info:
        models_info[name] = 0
    if num is None:
        models_info[name] += 1
    elif num >= models_info[name]:
        models_info[name] = num + 1
    return models_info[name] - 1


def init_models_info(folder_path):
    models_info = dict()
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    files = [f for f in os.listdir(folder_path) if re.search(r'.+_[0-9]+\.pth$', f)]
    for f in files:
        pre, ext = os.path.splitext(f)
        a = pre.split('_')
        name = '_'.join(a[:-1])
        num = int(a[-1])
        update_models_info(models_info, name, num)
    return models_info


def train_benign_models(
        n: int,
        folder_path: str,
        model_name_candidates: list[str] = ['resnet18_comp'],
        **kwargs):

    if kwargs['seed'] is None:
        dec, _ = math.modf(time.time())
        seed = int(dec*1000000000)%(999997)
        set_seed(seed)

    # folder_path = kwargs['folder_path']
    models_info = init_models_info(folder_path)

    '''
    files = [f for f in os.listdir(folder_path) if re.search(r'.+_[0-9]+\.pth$', f)]
    for f in files:
        if f.startswith('resnet'): continue
        print(f, models_info['resnet18_comp'])


        num = update_models_info(models_info, 'resnet18_comp')
        print(num)
        cmmd = 'mv '+os.path.join(folder_path, f)+' '+os.path.join(folder_path, f'resnet18_comp_{num}.pth')
        print(cmmd)

        os.system(cmmd)

    exit(0)
    # '''

    n_candi = len(model_name_candidates)
    model_name_list = np.repeat(model_name_candidates, n//n_candi)
    if len(model_name_list) < n:
        rest = np.random.choice(model_name_candidates, n - len(model_name_list))
        model_name_list = np.concatenate([model_name_list, rest], axis=0)
    np.random.shuffle(model_name_list)
    print(model_name_list)

    for model_name in model_name_list:
        num = update_models_info(models_info, model_name)
        save_name = f'{model_name}_{num}.pth'
        file_path = os.path.join(folder_path, save_name)

        kwargs['seed'] = (random.randint(1, 1000000)*1234567)%(999997)
        env = trojanvision.environ.create(**kwargs)
        dataset = trojanvision.datasets.create(**kwargs)
        kwargs['model_name'] = model_name
        model = trojanvision.models.create(dataset=dataset, **kwargs)
        trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)

        if env['verbose']:
            trojanvision.summary(env=env, dataset=dataset, model=model, trainer=trainer)
        model._train(**trainer, file_path=file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    kwargs = parser.parse_args().__dict__

    n = 99
    dataset_name = kwargs['dataset_name']
    # model_name_candidates = ['resnet18_comp', 'vgg13_bn', 'shufflenetv2_x1_0_comp', 'mobilenet_v2_comp']
    model_name_candidates = ['resnet50', 'vgg16_bn', 'shufflenetv2_x2_0', 'mobilenet_v3_large']
    train_benign_models(n, f'./benign_{dataset_name}', model_name_candidates, **kwargs)
