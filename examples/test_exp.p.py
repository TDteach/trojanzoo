import trojanvision
import argparse
import os
import re
import time
import math

import random
import numpy as np
import torch
from tqdm import tqdm


def get_inter_data(dataset):
    for data in dataset.loader['train']:
        data0 = data
        break
    X = dataset.get_data(data0)[0]
    x_list = list()
    for i in range(100):
        inter = torch.rand([len(X), 1, 1, 1], device='cuda')
        x_list.append((X * inter).detach().cpu().numpy())
    x_numpy = np.asarray(x_list)
    with open('inter_x.npy', 'wb') as f:
        np.save(f, x_numpy)
    print('inter data saved to inter_x.npy')

def get_inter_info(dataset, inter_numpy):

    info_list = list()
    with torch.no_grad():
        for z in tqdm(inter_numpy):
            z = torch.from_numpy(z).to('cuda')
            dis_list = list()

            st_time = time.time()
            for data in dataset.loader['train']:
                x = dataset.get_data(data)[0]
                x = torch.reshape(x, (len(x), -1))
                dis = x-z
                dis = torch.norm(dis, dim=-1)
                dis_list.append(dis.detach().cpu().numpy())
            dis_list = np.concatenate(dis_list, axis=0)
            ed_time = time.time()
            # print(ed_time - st_time)
            # print(len(dis_list))
            info = {
                'min': np.min(dis_list),
                'avg': np.mean(dis_list),
                'exp': np.mean(np.exp(dis_list)),
            }
            info_list.append([info['min'], info['avg'], info['exp']])

    with open("inter_info.npy", 'wb') as f:
        np.save(f, np.asarray(info_list))

def get_inter_probs(dataset, inter_x, folder_path):
    files = [f for f in os.listdir(folder_path) if re.search(r'.+\.pth$', f)]
    print(files)
    rst_list = list()
    for f in tqdm(files):
        print(f)
        pre, ext = os.path.splitext(f)
        a = pre.split('_')
        if re.search(r'^[0-9]+', a[-1]):
            model_name = '_'.join(a[:-1])
        else:
            model_name = pre
        kwargs['model_name'] = model_name
        model = trojanvision.models.create(dataset=dataset, **kwargs)
        path = os.path.join(folder_path, f)
        model.load(path)

        probs_list = list()
        model.eval()
        with torch.no_grad():
            for x in inter_x:
                x = torch.from_numpy(x).to('cuda')
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                probs_list.append(probs.detach().cpu().numpy())
        rst_list.append(probs_list)

    with open('probs_list.npy', 'wb') as f:
        np.save(f, np.asarray(rst_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    kwargs = parser.parse_args().__dict__

    env = trojanvision.environ.create(**kwargs)

    '''
    dataset = trojanvision.datasets.create(**kwargs)
    get_inter_data(dataset)
    # '''

    '''
    with open('inter_x.npy', 'rb') as f:
        inter_x = np.load(f)
    z = np.concatenate(inter_x, axis=0)
    print(z.shape)
    zz = np.reshape(z, (len(z), -1))
    print(zz.shape)
    dataset = trojanvision.datasets.create(**kwargs)
    get_inter_info(dataset, zz)
    # '''

    '''
    with open('inter_x.npy', 'rb') as f:
        inter_x = np.load(f)
    dataset = trojanvision.datasets.create(**kwargs)
    print(dataset.name)
    folder_path = f'./data/model/image/{dataset.name}'
    get_inter_probs(dataset, inter_x, folder_path)
    # '''

