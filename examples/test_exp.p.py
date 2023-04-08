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


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')

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
    files = sorted(files)
    print(files)
    rst_list = list()
    for f in tqdm(files):

        if not f.startswith('resnet18_comp'): continue

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
        rst_list.append(np.concatenate(probs_list, axis=0))
        del model

    with open('probs_list.npy', 'wb') as f:
        np.save(f, np.asarray(rst_list))


def main():
    with open('probs_list.npy', 'rb') as f:
        data = np.load(f)


    with open('inter_info.npy', 'rb') as f:
        inter_info = np.load(f)


    data = np.transpose(data, (1, 2, 0))
    std_mat = np.std(data, axis=-1)
    max_std = np.max(std_mat, axis=-1)
    min_std = np.min(std_mat, axis=-1)
    print(std_mat.shape)
    print(std_mat[0])
    print(max_std)
    print(max(max_std), min(max_std))

    hist, bins = np.histogram(max_std, bins=100)
    print(hist)
    print(bins)
    print(bins[1]-bins[0])
    print(bins[2]-bins[1])

    import matplotlib.pyplot as plt
    # _ = plt.hist(min_std, bins=100)


    # a = inter_info[:,2]
    # a = np.log(a)
    a = inter_info[:,0]
    order = np.argsort(a)

    x, y = a[order], max_std[order]

    mm = np.polyfit(x, y, 6)
    print(mm)
    yy = np.polyval(mm, x)



    plt.plot(x, y, '.')
    plt.plot(x, yy)
    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    kwargs = parser.parse_args().__dict__

    env = trojanvision.environ.create(**kwargs)

    '''
    print(model_name_list)
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

    # folder_path = f'./data/model/image/{dataset.name}'
    folder_path = f'./benign_{dataset.name}'
    get_inter_probs(dataset, inter_x, folder_path)
    # '''

    main()

