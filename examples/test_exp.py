import trojanvision
import argparse
import os
import re
import time
import math

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import pickle

from scipy.special import logsumexp

import torch
import torch.nn.functional as F

from gsw.gswnn import GSW_NN
from gsw.mlp import MLP


def savitzky_golay(y, window_size, order, deriv=0, rate=1, return_err=False):
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
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    b = b.astype(np.float64)
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    ry = np.convolve(m[::-1], y, mode='valid')
    if not return_err:
        return ry

    err_list = list()
    for i in range(len(ry)):
        ll, rr = max(0,i-half_window), min(len(y), i+half_window+1)
        err = np.mean((y[ll:rr] - ry[i])**2)
        err_list.append(np.sqrt(err))
    err_list = np.asarray(err_list)
    return ry, err_list


def get_GMM_for_model(dataset, model, covariance_type='tied'):
    name = dataset.name

    fm_list = list()
    for data in dataset.loader['train']:
        x = dataset.get_data(data)[0]
        final_fm = model.get_final_fm(x)
        fm_list.append(final_fm.detach().cpu().numpy())
    fm_list = np.concatenate(fm_list, axis=0)
    print(fm_list.shape, model.num_classes)

    X = fm_list
    from sklearn.mixture import GaussianMixture
    # gm = GaussianMixture(n_components=model.num_classes, random_state=0, max_iter=500, verbose=2, tol=1e-4, covariance_type='diag').fit(X)
    gm = GaussianMixture(n_components=model.num_classes, random_state=0, max_iter=500, verbose=2, tol=1e-4, covariance_type=covariance_type).fit(X)
    print(gm.means_.shape)

    return gm




def get_inter_data(dataset):
    name = dataset.name

    x_list = list()
    for data in dataset.loader['train']:
        x = dataset.get_data(data)[0]
        anchors = torch.rand(x.shape, device='cuda')
        inter = torch.rand([len(x), 1, 1, 1], device='cuda')
        nx = anchors * (1-inter) + x * inter
        x_list.append(nx.detach().cpu().numpy())
    pp = f'{name}_inter_x.pkl'
    with open(pp, 'wb') as f:
        pickle.dump(x_list, f)
    print('inter data saved to', pp)

def get_inter_info(dataset, inter_numpy):
    name = dataset.name

    print('gather all data')
    st_time = time.time()
    a = list()
    for data in dataset.loader['train']:
        x = dataset.get_data(data)[0]
        x = torch.reshape(x, (len(x), -1))
        a.append(x.detach().cpu().numpy())
    data = np.concatenate(a, axis=0)
    ed_time = time.time()
    print(ed_time - st_time)

    from fast_laplacian_kde import kde

    bandwidth = 1

    rst = list()
    for z in tqdm(inter_numpy):
        p = kde(z, data, bandwidth)
        try:
            lp = math.log(p)
        except:
            lp = -np.inf
        print(p, lp)
        rst.append(-lp)
    rst = np.reshape(rst, [len(rst), 1])
    print(rst.shape)

    '''
    with open("inter_info.npy", 'rb') as f:
        inter_info = np.load(f)

    inter_info = np.concatenate([inter_info, rst], axis=1)
    print(inter_info.shape)
    '''

    with open(f"{name}_inter_info.npy", 'wb') as f:
        np.save(f, np.asarray(rst))
    exit(0)

    info_list = list()
    with torch.no_grad():
        for z in tqdm(inter_numpy):
            z = torch.from_numpy(z).to('cuda')
            dis_list = list()

            for data in dataset.loader['train']:
                x = dataset.get_data(data)[0]
                x = torch.reshape(x, (len(x), -1))
                dis = x-z
                dis = torch.norm(dis, dim=-1)
                dis_list.append(dis.detach().cpu().numpy())
            dis_list = np.concatenate(dis_list, axis=0)
            info = {
                'min': np.min(dis_list),
                'avg': np.mean(dis_list),
                'exp': np.mean(np.exp(dis_list)),
            }
            info_list.append([info['min'], info['avg'], info['exp']])

    with open("inter_info.npy", 'wb') as f:
        np.save(f, np.asarray(info_list))



def get_model_name_from_path(f):
    pre, ext = os.path.splitext(f)
    a = pre.split('_')
    if re.search(r'^[0-9]+', a[-1]):
        model_name = '_'.join(a[:-1])
    else:
        model_name = pre

    return model_name

def load_model_from_path(f, folder_path, dataset, kwargs):
    model_name = get_model_name_from_path(f)
    kwargs['model_name'] = model_name
    model = trojanvision.models.create(dataset=dataset, **kwargs)
    path = os.path.join(folder_path, f)
    model.load(path)
    return model


def get_GMM_models(dataset, folder_path, cov_type='tied'):
    name = dataset.name

    files = [f for f in os.listdir(folder_path) if re.search(r'.+\.pth$', f)]
    files = sorted(files)
    for f in tqdm(files):
        model = load_model_from_path(f, folder_path, dataset, kwargs)
        model.eval()
        with torch.no_grad():
            gm = get_GMM_for_model(dataset, model, cov_type)
        pp = f'{cov_type}_{f}_gm.pkl'
        with open(pp, 'wb') as fh:
            pickle.dump(gm, fh)

        del model





def get_inter_probs(dataset, inter_x, folder_path):
    name = dataset.name

    files = [f for f in os.listdir(folder_path) if re.search(r'.+\.pth$', f)]
    files = sorted(files)
    rst_list = list()
    for f in tqdm(files):

        # if not f.startswith('resnet18_comp'): continue

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
                probs = model.get_prob(x)
                probs_list.append(probs.detach().cpu().numpy())
        rst_list.append(np.concatenate(probs_list, axis=0))
        del model

    with open(f'{name}_probs_list.npy', 'wb') as f:
        np.save(f, np.asarray(rst_list))



def get_inter_GMM_info(dataset, inter_x, folder_path, cov_type='tied'):
    name = dataset.name

    files = [f for f in os.listdir(folder_path) if re.search(r'.+\.pth$', f)]
    files = sorted(files)
    rst_list = list()
    for f in tqdm(files):

        # if not f.startswith('resnet18_comp'): continue

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

        pp = f'{name}_{cov_type}_gm/{cov_type}_{f}_gm.pkl'
        try:
            with open(pp, 'rb') as fh:
                gm = pickle.load(fh)
        except:
            continue

        print(f)

        probs_list = list()
        model.eval()
        with torch.no_grad():
            for x in inter_x:
                x = torch.from_numpy(x).to('cuda')
                final_fm = model.get_final_fm(x).detach().cpu().numpy()
                probs = gm.score_samples(final_fm)
                # print(np.max(probs), np.min(probs))
                # print(probs.shape)
                probs_list.append(probs)
        rst_list.append(np.concatenate(probs_list, axis=0))
        del model

    with open(f"{name}_inter_{cov_type}_GMM_info.npy", 'wb') as f:
        np.save(f, np.asarray(rst_list))




def fill_nan(a):
    for i in range(1, len(a)):
        if np.isnan(a[i]):
            a[i] = a[i-1]
    return a


def remove_outliers(x, y, n_bins=50, keep_ratio='std', min_ins=2):
    yyy, xxx, bid = stats.binned_statistic(x, y, 'mean', bins=n_bins)
    del_list = list()
    idx = np.arange(len(x))
    for i in range(n_bins):
        s = bid == (i+1)
        jj = idx[s]
        if np.sum(s) < min_ins: continue
        zy = y[s]
        zz = np.abs(zy - yyy[i])
        zz = np.sort(zz)
        if keep_ratio=='std':
            thr = np.std(zy)
        else:
            thr = zz[int(len(zz)*keep_ratio)]
        for j in jj:
            if abs(y[j]-yyy[i]) > thr:
                del_list.append(j)
    nx, ny = list(), list()
    del_list = sorted(del_list)
    print('remove {} outliers'.format(len(del_list)))
    for i, (x_, y_) in enumerate(zip(x,y)):
        if len(del_list) > 0 and i == del_list[0]:
            del_list.pop(0)
            continue
        nx.append(x_)
        ny.append(y_)

    return nx, ny


def remove_inf_nan(x, y):
    nx, ny = list(), list()
    for _x, _y in zip(x,y):
        if any([np.isinf(_x), np.isinf(_y), np.isnan(_x), np.isnan(_y)]):
            continue
        nx.append(_x)
        ny.append(_y)
    return np.asarray(nx), np.asarray(ny)


def wasserstein_between_two_gm(X1, X2, gm1, gm2):
    trans = MLP(din=512, dout=512, num_filters=512, depth=1)
    # w = trans.features.linear01.weight
    optim = torch.optim.Adam(trans.parameters(), lr=1e-3, betas=(0.5,0.99), weight_decay=1e-5)
    trans.cuda()

    X, Y = X1, X2

    for k in range(200):
        optim.zero_grad()
        nX = trans(X)

        loss = F.mse_loss(nX, Y)

        loss.backward()
        optim.step()

    XX, _ = gm1.sample(10000)
    YY, _ = gm2.sample(10000)
    XX = torch.from_numpy(XX).float().cuda()
    YY = torch.from_numpy(YY).float().cuda()
    nXX = trans(XX)

    w_model = GSW_NN(din=512, nofprojections=1, num_filters=32, model_depth=1)
    w_model.model = w_model.model.double()

    dist = w_model.max_gsw(nXX.double().data, YY.double().data, iterations=10000)

    del w_model

    return dist.sqrt().item(), trans

def get_fm_from_loader(model, loader, get_data_fn):
    fm_list = list()
    with torch.no_grad():
        for data in loader:
            x = get_data_fn(data)[0]
            final_fm = model.get_final_fm(x)
            fm_list.append(final_fm.detach().cpu().numpy())
    return np.concatenate(fm_list, axis=0)




def compare_infos(dataset, folder_path, cov_type='tied'):
    name = dataset.name
    gmm_folder = f'{name}_{cov_type}_gm'
    files = os.listdir(gmm_folder)
    model_paths = dict()
    for f in files:
        if not f.endswith('_gm.pkl'):
            continue
        p = '_'.join(f.split('_')[1:-1])
        md = p.split('_')[0]
        if md not in model_paths:
            model_paths[md] = list()
        model_paths[md].append(p)


    all_fm = dict()
    for md, p_list in model_paths.items():
        if not md.startswith('resnet'): continue
        all_fm[md] = list()
        n = len(p_list)
        for i in range(n):
            print(p_list[i])
            model = load_model_from_path(p_list[i], folder_path, dataset, kwargs)
            model.eval()

            #for na, v in model._model.classifier.named_parameters():
            #    print(na, v.shape)
            #exit(0)

            valid_fm = get_fm_from_loader(model, dataset.loader['valid'], dataset.get_data)

            new_dataset = dataset.get_org_dataset('train',transform=dataset.get_transform(mode='valid'))
            loader = dataset.get_dataloader('train', dataset=new_dataset, shuffle=False)
            train_fm = get_fm_from_loader(model, loader, dataset.get_data)

            f = p_list[i]
            pp = os.path.join(f'{name}_{cov_type}_gm', f'{cov_type}_{f}_gm.pkl')
            all_fm[md].append((valid_fm, train_fm, pp))
            del model


    rst_dict = dict()
    for md, fm_list in all_fm.items():
        if not md.startswith('resnet'): continue

        rst_dict[md] = dict()

        n = len(fm_list)
        for i in range(n):
            valid_fm, train_fm, pp = fm_list[i]
            X = torch.from_numpy(valid_fm).float().cuda()
            XX = torch.from_numpy(train_fm).float().cuda()
            with open(pp, 'rb') as fh:
                gm = pickle.load(fh)

            for j in range(i+1, n):
                _valid_fm, _train_fm, _pp = fm_list[j]

                Y = torch.from_numpy(_valid_fm).float().cuda()
                YY = torch.from_numpy(_train_fm).float().cuda()
                with open(_pp, 'rb') as fh:
                    _gm = pickle.load(fh)

                dist, trans = wasserstein_between_two_gm(X, Y, gm, _gm)
                print(i, j, dist)

                trans.eval()
                trans.cpu()
                rst = {
                    'trans': trans.state_dict(),
                    'dist': dist.sqrt().item(),
                }

                rst_dict[md][(pp, _pp)] = rst

    with open('benign_models_between.pkl','wb') as fh:
        pickle.dump(rst_dict,fh)

    exit(0)


    name = dataset.name

    '''
    fm_list = list()
                dist = w_model.max_gsw(nXX.data, YY.data, iterations=10000)
                print(i, j, dist)

                trans.eval()
                trans.cpu()
                rst = {
                    'trans': trans.state_dict(),
                    'dist': dist,
                }

                rst_dict[md][(pp, _pp)] = rst

    with open('benign_models_between.pkl','wb') as fh:
        pickle.dump(rst_dict,fh)

    exit(0)


    name = dataset.name

    '''
    fm_list = list()
    for data in dataset.loader['train']:
        x = dataset.get_data(data)[0]
        final_fm = model.get_final_fm(x)
        fm_list.append(final_fm.detach().cpu().numpy())
    fm_list = np.concatenate(fm_list, axis=0)
    print(fm_list.shape, model.num_classes)
    # '''



    data = np.transpose(data, (1, 0))

    n, m = data.shape[:2]
    #for j in range(m):
    #    data[:, j] -= logsumexp(data[:,j])

    a = []
    for j1 in tqdm(range(m)):
        logsum = logsumexp(data[:,j1])
        for j2 in range(j1+1, m):
            dif = []
            for i in range(n):
                if data[i,j1] <= data[i,j2]:
                    dif.append(-np.inf)
                    continue
                # d = logsumexp([data[i,j1], data[i,j2]], b=[1,-1])
                d = data[i,j1] - data[i,j2]
                # print(data[i,j1], data[i,j2], d)
                dif.append(d)
            surpus = logsumexp(dif)-logsum
            print(j2, surpus)
            a.append(surpus)

    _ = plt.hist(a, bins='auto')
    plt.show()




def draw_gmm_results(dataset_name, cov_type='tied'):
    name = dataset_name
    with open(f'{name}_inter_{cov_type}_GMM_info.npy', 'rb') as f:
        data = np.load(f)

    with open(f'{name}_inter_info.npy', 'rb') as f:
        inter_info = np.load(f)

    a = inter_info[:,0]
    data = np.transpose(data, (1, 0))

    if False:
        a = -a-logsumexp(-a)
        n, m = data.shape[:2]
        for j in range(m):
            data[:, j] -= logsumexp(data[:,j])
        mean_ary = []
        for i in range(n):
            avg = logsumexp(data[i,:])-np.log(m)
            mean_ary.append(avg)

        x, y = a, mean_ary
        x, y = remove_inf_nan(x, y)

        order = np.argsort(x)
        x, y = x[order], y[order]

        x, y = remove_outliers(x, y, n_bins=100)
    else:
        mean_ary = np.mean(data, axis=-1)
        x, y = a, -mean_ary
        x, y = remove_inf_nan(x, y)

        order = np.argsort(x)
        x, y = x[order], y[order]

        x, y = remove_outliers(x, y, n_bins=100)
        x = (x-np.min(x))/(np.max(x)-np.min(x))
        y = (y-np.min(y))/(np.max(y)-np.min(y))


    n_bins = 200
    yy, xx, _ = stats.binned_statistic(x, y, 'mean', bins=n_bins)
    ss, _, _ = stats.binned_statistic(x, y, 'std', bins=n_bins)
    yy = fill_nan(yy)
    ss = fill_nan(ss)
    xxx = np.asarray([(xx[i]+xx[i+1])/2 for i in range(len(xx)-1)])

    ws = 21
    yyy = savitzky_golay(yy, window_size=ws, order=1)
    plt.plot(x, y, '.')
    plt.plot(xxx, yy)
    intl = 7
    plt.errorbar(xxx[::intl],yyy[::intl],ss[::intl],marker='^')

    plt.show()






def main(dataset_name):
    name = dataset_name
    with open(f'{name}_probs_list.npy', 'rb') as f:
        data = np.load(f)

    with open(f'{name}_inter_info.npy', 'rb') as f:
        inter_info = np.load(f)

    print(inter_info.shape)

    data = np.transpose(data, (1, 2, 0))
    std_mat = np.std(data, axis=-1)
    mean_mat = np.mean(data, axis=-1)
    max_std = np.max(std_mat, axis=-1)
    min_std = np.min(std_mat, axis=-1)
    mean_std = np.mean(std_mat, axis=-1)
    a_std = std_mat[:, 3]
    print(max(max_std), min(max_std))

    '''
    a = np.repeat(inter_info, std_mat.shape[1], axis=1)
    b = np.log(mean_mat)
    x = (a-b).flatten()
    y = std_mat.flatten()
    '''

    a = inter_info[:,0]
    # a = np.log(a)

    # x, y = a, max_std
    x, y = a, mean_std

    x, y = remove_inf_nan(x, y)

    order = np.argsort(x)
    x, y = x[order], y[order]

    # x, y = remove_outliers(x, y, n_bins=100, keep_ratio=0.9)
    x, y = remove_outliers(x, y, n_bins=100)

    # mm = np.polyfit(x, y, 6)
    # yy = np.polyval(mm, x)
    # yy = savitzky_golay(y, window_size=1007, order=2)



    '''
    ws = 501
    yyy, err = savitzky_golay(y, window_size=ws, order=1, return_err=True)
    plt.plot(x, y, '.')
    plt.plot(x, yyy)
    plt.errorbar(x[::200],yyy[::200],err[::200],marker='^')
    # '''

    # '''
    n_bins = 200
    yy, xx, _ = stats.binned_statistic(x, y, 'mean', bins=n_bins)
    ss, _, _ = stats.binned_statistic(x, y, 'std', bins=n_bins)
    yy = fill_nan(yy)
    ss = fill_nan(ss)
    xxx = np.asarray([(xx[i]+xx[i+1])/2 for i in range(len(xx)-1)])

    ws = 21
    yyy = savitzky_golay(yy, window_size=ws, order=1)
    plt.plot(x, y, '.')
    plt.plot(xxx, yy)
    intl = 7
    plt.errorbar(xxx[::intl],yyy[::intl],ss[::intl],marker='^')
    # '''

    plt.show()


def compare_benign_trojan(dataset, cov_type='tied'):
    name = dataset.name

    benign_folder = f'benign_{name}'
    benign_gm_folder = f'{name}_{cov_type}_gm'
    trojan_folder = f'trojan_{name}'
    trojan_gm_folder = f'{name}_trojan_{cov_type}_gm'

    files = os.listdir(trojan_gm_folder)
    model_paths = dict()
    for f in files:
        if not f.endswith('_gm.pkl'):
            continue
        if not os.path.exists(os.path.join(benign_gm_folder, f)):
            continue
        p = '_'.join(f.split('_')[1:-1])
        md = p.split('_')[0]
        if md not in model_paths:
            model_paths[md] = list()
        model_paths[md].append(p)


    for md, p_list in model_paths.items():
        if not md.startswith('resnet'): continue

        n = len(p_list)
        for i in range(n):
            f = p_list[i]

            benign_model = load_model_from_path(f, benign_folder, dataset, kwargs)
            benign_model.eval()
            valid_fm_benign = get_fm_from_loader(benign_model, dataset.loader['valid'], dataset.get_data)
            del benign_model

            trojan_model = load_model_from_path(f, trojan_folder, dataset, kwargs)
            trojan_model.eval()
            valid_fm_trojan = get_fm_from_loader(trojan_model, dataset.loader['valid'], dataset.get_data)
            del trojan_model

            bpp = os.path.join(benign_gm_folder, f'{cov_type}_{f}_gm.pkl')
            with open(bpp, 'rb') as fh:
                bgm = pickle.load(fh)
            tpp = os.path.join(trojan_gm_folder, f'{cov_type}_{f}_gm.pkl')
            with open(tpp, 'rb') as fh:
                tgm = pickle.load(fh)

            dist, trans = wasserstein_between_two_gm(valid_fm_benign, valid_fm_trojan, bgm, tgm)
            print(dist)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    kwargs = parser.parse_args().__dict__

    env = trojanvision.environ.create(**kwargs)

    '''
    dataset = trojanvision.datasets.create(**kwargs)
    print('get inter data for', dataset.name)
    get_inter_data(dataset)
    # '''

    '''
    dataset = trojanvision.datasets.create(**kwargs)
    print('get inter info for', dataset.name)
    name = dataset.name
    with open(f'{name}_inter_x.pkl', 'rb') as f:
        inter_x = pickle.load(f)
    z = np.concatenate(inter_x, axis=0)
    print(z.shape)
    zz = np.reshape(z, (len(z), -1))
    print(zz.shape)
    get_inter_info(dataset, zz)
    # '''

    '''
    dataset = trojanvision.datasets.create(**kwargs)
    print('get GMM models for', dataset.name)
    name = dataset.name
    # folder_path = f'./benign_{name}'
    folder_path = f'./trojan_{name}'
    get_GMM_models(dataset, folder_path)
    # '''

    '''
    dataset = trojanvision.datasets.create(**kwargs)
    print('get inter GMM info for', dataset.name)
    name = dataset.name
    with open(f'{name}_inter_x.pkl', 'rb') as f:
        inter_x = pickle.load(f)
    print(dataset.name)

    folder_path = f'./benign_{name}'
    get_inter_GMM_info(dataset, inter_x, folder_path)
    # '''

    '''
    dataset = trojanvision.datasets.create(**kwargs)
    print('get inter probs for', dataset.name)
    name = dataset.name
    with open(f'{name}_inter_x.pkl', 'rb') as f:
        inter_x = pickle.load(f)
    print(dataset.name)

    # folder_path = f'./data/model/image/{dataset.name}'
    folder_path = f'./benign_{name}'
    get_inter_probs(dataset, inter_x, folder_path)
    # '''

    '''
    dataset = trojanvision.datasets.create(**kwargs)
    print('draw GMM figure for', dataset.name)
    draw_gmm_results(dataset.name)
    # '''

    # '''
    kwargs['valid_batch_size'] = kwargs['batch_size']
    dataset = trojanvision.datasets.create(**kwargs)
    print('compare GMM info', dataset.name)
    name = dataset.name
    folder_path = f'./benign_{name}'
    compare_infos(dataset, folder_path)
    # '''


    '''
    dataset = trojanvision.datasets.create(**kwargs)
    print('draw figure for', dataset.name)
    main(dataset.name)
    # '''

