import numpy as np
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import torch

def plot_rgb_cloud(cloud, ax):
    colors = np.clip(cloud, 0, 1)
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=colors)
    ax.set_xlabel('Red'); ax.set_ylabel('Green'); ax.set_zlabel('Blue');

def plot_combined_generated(Dec, Z, D, inv_transform=None, show=True):
    Dec.train(False)
    
    if inv_transform is None:
        inv_transform = lambda x: x
    
    fig, axes = plt.subplots(2, len(Z), figsize=(2 * len(Z), 4))
    
    Dec_Z = inv_transform(Dec(Z).permute(0, 2, 3, 1).cpu().detach().numpy()).clip(0,1)
    axes.flatten()[0].set_ylabel('Dec(Z)', fontsize=15)
    for i in range(len(Z)):
        axes.flatten()[i].imshow(Dec_Z[i], cmap='gray')
    
    D.train(False)
    D_Z = D.push(Z.requires_grad_(True))
    D.train(True)

    Dec_D_Z = inv_transform(Dec(D_Z).permute(0, 2, 3, 1).cpu().detach().numpy()).clip(0,1)
    axes[1, 0].set_ylabel('Dec of\nD.push(Z)', fontsize=15)
    for i in range(len(Z)):
        axes[1, i].imshow(Dec_D_Z[i], cmap='gray')
     
    for ax in axes.flatten():
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
    
    if show:
        plt.show()
        
    return fig, axes

def plot_latent_pca(Z, E, D, pca, n_pairs=3, show=True):
    assert n_pairs > 1
    D.train(False);
    pca_Z = pca.transform(Z.cpu().detach().numpy().reshape(len(Z), -1))
    pca_E = pca.transform(E.cpu().detach().numpy().reshape(len(E), -1))
    D_Z = D.push(Z).cpu().detach().numpy().reshape(len(Z), -1)
    pca_D_Z = pca.transform(D_Z)
    
    fig, axes = plt.subplots(n_pairs, 3, figsize=(12, 4 * n_pairs), sharex=True, sharey=True)
    
    for n in range(n_pairs):
        axes[n, 0].set_ylabel(f'Component {2*n+1}')
        axes[n, 0].set_xlabel(f'Component {2*n}')
        axes[n, 0].set_title(f'Initial Z', fontsize=15)
        axes[n, 1].set_title('Transported Z', fontsize=15)
        axes[n, 2].set_title('Latent Space', fontsize=15)

        axes[n, 0].scatter(pca_Z[:, 2*n], pca_Z[:, 2*n+1], color='b', alpha=0.5)
        axes[n, 1].scatter(pca_D_Z[:, 2*n], pca_D_Z[:, 2*n+1], color='r', alpha=0.5)
        axes[n, 2].scatter(pca_E[:, 2*n], pca_E[:, 2*n+1], color='g', alpha=0.5)
        
    fig.tight_layout()
    D.train(True)
    
    if show:
        plt.show()
    return fig, axes