import random
import torch
import numpy as np
from scipy.linalg import sqrtm
import sklearn.datasets
from sklearn.mixture import GaussianMixture

class Sampler:
    def __init__(
        self, device='cuda',
        dtype=torch.float,
        requires_grad=False
    ):
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad
    
    def sample(self, batch_size=5):
        pass
    
class SwissRollSampler(Sampler):
    def __init__(
        self, dim=2, device='cuda',
        dtype=torch.float, requires_grad=False
    ):
        super(SwissRollSampler, self).__init__(
            device=device, dtype=dtype, requires_grad=requires_grad
        )
        assert dim == 2
        self.dim = 2
        
    def sample(self, batch_size=10):
        batch = sklearn.datasets.make_swiss_roll(
            n_samples=batch_size,
            noise=0.8
        )[0].astype('float32')[:, [0, 2]] / 7.5
        return torch.tensor(
            batch, dtype=self.dtype,
            device=self.device, requires_grad=self.requires_grad
        )
    
class StandartNormalSampler(Sampler):
    def __init__(
        self, dim=1, device='cuda',
        dtype=torch.float, requires_grad=False
    ):
        super(StandartNormalSampler, self).__init__(
            device=device, dtype=dtype, requires_grad=requires_grad
        )
        self.dim = dim
        
    def sample(self, batch_size=10):
        return torch.randn(
            batch_size, self.dim, dtype=self.dtype,
            device=self.device, requires_grad=self.requires_grad
        )
    
class StandartUniformSampler(Sampler):
    def __init__(
        self, dim=1, device='cuda',
        dtype=torch.float, requires_grad=False
    ):
        super(StandartUniformSampler, self).__init__(
            device=device, dtype=dtype, requires_grad=requires_grad
        )
        self.dim = dim
        
    def sample(self, batch_size=10):
        return torch.rand(
            batch_size, self.dim, dtype=self.dtype,
            device=self.device, requires_grad=self.requires_grad
        )
    
class BallUniformSampler(Sampler):
    def __init__(
        self, dim=1, device='cuda',
        dtype=torch.float, requires_grad=False
    ):
        super(BallUniformSampler, self).__init__(
            device=device, dtype=dtype, requires_grad=requires_grad
        )
        self.dim = dim
        
    def sample(self, batch_size=10):
        batch = torch.randn(
            batch_size, self.dim,
            device=self.device, dtype=self.dtype
        )
        batch /= torch.norm(batch, dim=1)[:, None]
        r = torch.rand(
            batch_size, device=self.device, dtype=self.dtype
        ) ** (1. / self.dim)
        return torch.tensor(
            (batch.transpose(0, 1) * r).transpose(0, 1),
            device=self.device, dtype=self.dtype,
            requires_grad=self.requires_grad
        )
    
class Mix8GaussiansSampler(Sampler):
    def __init__(
        self, with_central=False, std=1, r=12, dim=2, device='cuda',
        dtype=torch.float, requires_grad=False
    ):
        super(Mix8GaussiansSampler, self).__init__(
            device=device, dtype=dtype, requires_grad=requires_grad
        )
        assert dim == 2
        self.dim = 2
        self.std, self.r = std, r
        
        self.with_central = with_central
        centers = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        if self.with_central:
            centers.append((0, 0))
        self.centers = torch.tensor(
            centers, device=self.device, dtype=self.dtype
        )
        
    def sample(self, batch_size=10):
        batch = torch.randn(
            batch_size, self.dim,
            device=self.device, dtype=self.dtype
        )
        indices = random.choices(range(len(self.centers)), k=batch_size)
        batch *= self.std
        batch += self.r * self.centers[indices, :]
        return torch.tensor(
            batch, device=self.device, dtype=self.dtype,
            requires_grad = self.requires_grad
        )
    
class MixN2GaussiansSampler(Sampler):
    def __init__(self, n=5, dim=2, std=1, step=9, device='cuda',
        dtype=torch.float, requires_grad=False
    ):
        super(MixN2GaussiansSampler, self).__init__(
            device=device, dtype=dtype, requires_grad=requires_grad
        )
        
        assert dim == 2
        self.dim = 2
        self.std, self.step = std, step
        
        self.n = n
        
        grid_1d = np.linspace(-(n-1) / 2., (n-1) / 2., n)
        xx, yy = np.meshgrid(grid_1d, grid_1d)
        centers = np.stack([xx, yy]).reshape(2, -1).T
        self.centers = torch.tensor(
            centers,
            device=self.device,
            dtype=self.dtype
        )
        
    def sample(self, batch_size=10):
        batch = torch.randn(
            batch_size, self.dim,
            device=self.device, dtype=self.dtype
        )
        indices = random.choices(range(len(self.centers)), k=batch_size)
        batch *= self.std
        batch += self.step * self.centers[indices, :]
        return torch.tensor(
            batch, device=self.device, dtype=self.dtype,
            requires_grad=self.requires_grad
        )
    
class TensorDatasetSampler(Sampler):
    def __init__(
        self, dataset, transform=None, storage='cpu', storage_dtype=torch.float, device='cuda',
        dtype=torch.float, requires_grad=False
    ):
        super(TensorDatasetSampler, self).__init__(
            device=device, dtype=dtype, requires_grad=requires_grad
        )
        self.storage = storage
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = lambda t: t
            
        self.storage_dtype = storage_dtype
        
        self.dataset = torch.tensor(
            dataset, device=storage, dtype=storage_dtype, requires_grad=False
        )

        
    def sample(self, batch_size=10):
        if batch_size:
            ind = random.choices(range(len(self.dataset)), k=batch_size)
        else:
            ind = range(len(self.dataset))
            
        batch = self.transform(torch.tensor(
            self.dataset[ind], device=self.device,
            dtype=self.dtype, requires_grad=False
        )).detach()
        if self.requires_grad:
            batch.requires_grad_(True)
        return batch
    

class NormalTensorDatasetSampler(Sampler):
    def __init__(
        self, dataset, device='cuda',
        dtype=torch.float, requires_grad=False,
        n_components=1
    ):
        super(NormalTensorDatasetSampler, self).__init__(
            device=device, dtype=dtype, requires_grad=requires_grad
        )
        self.sampler = GaussianMixture(n_components)
        self.sampler.fit(dataset)
        
    def sample(self, batch_size=10):
        if not batch_size:
            batch_size = 10
            
        batch = torch.tensor(
            self.sampler.sample(batch_size)[0], device=self.device,
            dtype=self.dtype, requires_grad=self.requires_grad
        )
        return batch

class Transformer(object):
    def __init__(
        self, device='cuda', dtype=torch.float,
        requires_grad=False
    ):
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad
        
class NormalNoiseTransformer(Transformer):
    def __init__(
        self, std=0.01,
        device='cuda', dtype=torch.float,
        requires_grad=False
    ):
        super(NormalNoiseTransformer, self).__init__(
            device=device, dtype=dtype,
            requires_grad=requires_grad
        )
        self.std = std
        
    def fit(self, base_sampler):
        self.base_sampler = base_sampler
        
    def sample(self, batch_size=4):
        batch = torch.tensor(
            self.base_sampler.sample(batch_size),
            device=self.device, requires_grad=False, dtype=self.dtype
        )
        batch = batch + self.std * torch.randn_like(batch)
        batch = batch.detach()
        batch.requires_grad_(self.requires_grad)
        return batch

class StandardNormalScaler(Transformer):
    def __init__(
        self, device='cuda',
        dtype=torch.float,
        requires_grad=False
    ):
        super(StandardNormalScaler, self).__init__(
            device=device, dtype=dtype,
            requires_grad=requires_grad
        )
        
    def fit(self, base_sampler, batch_size=1000):
        self.base_sampler = base_sampler
        batch = self.base_sampler.sample(batch_size).cpu().detach().numpy()
        
        mean, cov = np.mean(batch, axis=0), np.cov(batch.T)
        
        self.mean = torch.tensor(
            mean, device=self.device, dtype=self.dtype
        )
        
        multiplier = sqrtm(cov)
        self.multiplier = torch.tensor(
            multiplier, device=self.device, dtype=self.dtype
        )
        self.inv_multiplier = torch.tensor(
            np.linalg.inv(multiplier),
            device=self.device, dtype=self.dtype
        )
        torch.cuda.empty_cache()
        
    def sample(self, batch_size=10):
        batch = torch.tensor(
            self.base_sampler.sample(batch_size),
            device=self.device, dtype=self.dtype
        )
        batch -= self.mean
        batch @= self.inv_multiplier
        if self.requires_grad:
            batch.requires_grad_(True)
        return batch
    
    def inverse_transform(self, batch):
        batch @= self.multiplier
        batch += self.mean
        return batch
    
class LinearTransformer(Transformer):
    def __init__(
        self, weight, bias=None,
        device='cuda', dtype=torch.float,
        requires_grad=False
    ):
        super(LinearTransformer, self).__init__(
            device=device, dtype=dtype,
            requires_grad=requires_grad
        )
        
        self.weight = torch.tensor(weight, device=device, dtype=dtype, requires_grad=False)
        if bias is not None:
            self.bias = torch.tensor(bias, device=device, dtype=dtype, requires_grad=False)
        else:
            self.bias = None
        
    def fit(self, base_sampler):
        self.base_sampler = base_sampler
        
    def sample(self, batch_size=4):
        batch = torch.tensor(
            self.base_sampler.sample(batch_size),
            device=self.device, requires_grad=False, dtype=self.dtype
        )
        with torch.no_grad():
            batch = batch @ self.weight.T
            if self.bias is not None:
                batch += self.bias
        batch = batch.detach()
        batch.requires_grad_(self.requires_grad)
        return batch