import trojanvision
import argparse
import random
import os
import re


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
        a = pre.split('_')[-1]
        name = a[:-1].join('_')
        num = int(a[-1])
        update_models_info(models_info, name, num)
    return models_info


def train_benign_models(
        n: int,
        folder_path: str,
        model_name_candidates: list[str] = ['resnet18_comp'],
        **kwargs):
    # folder_path = kwargs['folder_path']
    models_info = init_models_info(folder_path)
    model_name_list = random.choices(model_name_candidates, k=n)
    for model_name in model_name_list:
        num = update_models_info(models_info, model_name)
        save_name = f'{model_name}_{num}.pth'
        file_path = os.path.join(folder_path, save_name)

        env = trojanvision.environ.create(**kwargs)
        dataset = trojanvision.datasets.create(**kwargs)
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

    n = 20
    dataset_name = kwargs['dataset_name']
    model_name_candidates = ['resnet18_comp', 'vgg13_bn', 'shufflenet2_comp', 'mobilenet_v2_comp']
    # model_name_candidates = ['resnet50', 'vgg16_bn', 'shufflenet2', 'mobilenet_v3_large']
    train_benign_models(n, f'./benign_{dataset_name}', model_name_candidates, **kwargs)
