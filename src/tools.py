import os, sys
import torchvision.datasets as datasets
import numpy as np
import pandas as pd
from tqdm import tqdm

import os, sys
import argparse
import collections
from scipy.io import savemat
from tqdm import trange
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import multiprocessing
import itertools

import torch
from PIL import Image
sys.path.append("..")

from .inception import InceptionV3
from .fid_score import get_activations_for_dataloader

import gc

def compute_l1_norm(model):
    regularizer = 0.
    for param in model.parameters():
        regularizer += torch.sum(torch.abs(param))
    return regularizer

def ewma(x, span=200):
    return pd.DataFrame({'x': x}).ewm(span=span).mean().values[:, 0]

def read_images_to_ram(path, mode='RGB', verbose=True):
    images = []
    fails = 0
    print('Reading images from {}'.format(path))
    for file in tqdm(os.listdir(path)) if verbose else os.listdir(path):
        try:
            with Image.open(os.path.join(path, file), 'r') as im:
                images.append(im.convert(mode).copy())
        except:
            fails += 1
            if verbose:
                print('Failed to read {}'.format(os.path.join(path, file)))
    print('{} succesful; {} fails'.format(len(images), fails)) if verbose else None
    return images

class ImageBatchSampler:
    def __init__(self, list_of_images, transform=None):
        self.list_of_images = list_of_images
        self.transform = transform if transform is not None else lambda x: x

    def sample(self, batch_size):
        idx = np.random.choice(range(len(self.list_of_images)), replace=True, size=batch_size)
        batch = [self.list_of_images[i] for i in idx]
        return torch.stack(list(map(self.transform, batch)))

def read_images(paths, mode='RGB', verbose=True):
    images = []
    for path in paths:
        try:
            with Image.open(path, 'r') as im:
                images.append(im.convert(mode).copy())
        except:
            if verbose:
                print('Failed to read {}'.format(path))
    return images
    
class ImagesReader:
    def __init__(self, mode='RGB', verbose=True):
        self.mode = mode
        self.verbose = verbose
        
    def __call__(self, paths):
        return read_images(paths, mode=self.mode, verbose=self.verbose)
    
def read_image_folder(path, mode='RGB', verbose=True, n_jobs=1):
    paths = [os.path.join(path, name) for name in os.listdir(path)]
    
    chunk_size = (len(paths) // n_jobs) + 1
    chunks = [paths[x:x+chunk_size] for x in range(0, len(paths), chunk_size)]
    
    pool = multiprocessing.Pool(n_jobs)
    
    chunk_reader = ImagesReader(mode, verbose)
    
    images = list(itertools.chain.from_iterable(
        pool.map(chunk_reader, chunks)
    ))
    pool.close()
    return images

def get_generated_inception_stats(G, Z_sampler, inv_transform, size, batch_size=16):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).cuda()
    model.eval()

    if batch_size > size:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = size

    pred_arr = np.empty((size, dims))

    for i in tqdm(range(0, size, batch_size)):
        start = i
        end = min(i + batch_size, size)
        
        G_Z = G(Z_sampler.sample(end-start).requires_grad_(True))
        if inv_transform is not None:
            G_Z = inv_transform(G_Z)
        batch = G_Z.detach().type(torch.FloatTensor).cuda()
        pred = model(batch)[0]

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    
    model = model.cpu()
    del model, pred_arr, pred, batch
    gc.collect()
    torch.cuda.empty_cache()
    
    return mu, sigma

def get_statistics_of_dataloader(dataloader, dims=2048, cuda=False, verbose=False):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()
        
    act = get_activations_for_dataloader(dataloader, model, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def energy_based_distance(X, Y, n_projections=10000, biased=False):
    '''
    An implementation of unbiased energy-based distance between
    two disributions given by i.i.d. batches.
    
    This implementation computes an unbiased sliced continuous
    ranking probability score (via random projections).
    It equals energy based distance up to a multiplicative
    constant depending on the dimension,
    see Theorem 4.1 of https://arxiv.org/pdf/1912.07048.pdf for details 
    '''
    assert X.size(1) == Y.size(1)
    
    thetas = torch.randn(n_projections, X.size(1)).cuda()
    thetas = thetas / thetas.norm(2, dim=1, keepdim=True)
    
    # Sorted projection of joint matrix and reverse sorted index
    pXY, idx = torch.sort(thetas @ torch.cat((X, Y), dim=0).transpose(0,1), dim=1)
    
    # Normalized indicator functions (1./X.size(0) for elements from X, -1./Y.size(0) for Y)
    I = torch.ones(idx.size(), dtype=torch.float32, device='cuda') / X.size(0)
    I[idx >= X.size(0)] = -1. / Y.size(0)
    
    SFXY = torch.cumsum(I, dim=1)
    scrps_biased = torch.mean(torch.sum((pXY[:, 1:] - pXY[:, :-1]) * SFXY[:, :-1] ** 2, dim=1))
    
    if biased:
        return scrps_biased
    
    pX_mask = idx < X.size(0)
    SFX = torch.cumsum(I[pX_mask].reshape(-1, X.size(0)), dim=1)
    pX = pXY[pX_mask].reshape(-1, X.size(0))
    var_SFX = torch.mean(torch.sum((pX[:, 1:] - pX[:, :-1]) * SFX[:, :-1] * (1. - SFX[:, :-1]), dim=1)) / (X.size(0) - 1)
    
    pY_mask = idx >= X.size(0)
    SFY = torch.cumsum(I[pY_mask].reshape(-1, Y.size(0)), dim=1)
    pY = pXY[pY_mask].reshape(-1, Y.size(0))
    var_SFY = torch.mean(torch.sum((pY[:, 1:] - pY[:, :-1]) * SFY[:, :-1] * (1. - SFY[:, :-1]), dim=1)) / (Y.size(0) - 1)
    
    return scrps_biased - var_SFX - var_SFY