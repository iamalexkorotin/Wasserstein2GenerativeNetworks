# Wasserstein-2 Generative Networks
This is the official `Python` implementation of the paper **Wasserstein-2 Generative Networks** (preprint on [arXiv](https://arxiv.org/abs/1909.13082)) by [Alexander Korotin](https://scholar.google.ru/citations?user=1rIIvjAAAAAJ&hl=en), [Vahe Egizarian](https://scholar.google.ru/citations?user=Bktg6JEAAAAJ&hl=en), [Arip Asadulaev](https://scholar.google.com/citations?user=wcdrgdYAAAAJ&hl=ru), [Alexander Safin](https://scholar.google.com/citations?user=ga3P-mAAAAAJ&hl=en) and [Evgeny Burnaev](https://scholar.google.ru/citations?user=pCRdcOwAAAAJ&hl=ru).

The repository contains reproducible `PyTorch` source code for computing **optimal transport maps** (and distances) in high dimensions via the **end-to-end non-minimax** method (proposed in the paper) by using **input convex neural networks**. Examples are provided for various real-world problems: color transfer, latent space mass transport, domain adaptation, style transfer.

<p align="center"><img src="pics/mappings.png" width="450" /></p>

## Prerequisites
The implementation is GPU-based. Single GPU (~GTX 1080 ti) is enough to run each particular experiment.
- [pytorch](http://pytorch.org/)
- [torchvision](https://github.com/pytorch/vision)
- CUDA + CuDNN

## Repository structure
All the experiments are issued in the form of pretty self-explanatory jupyter notebooks (`notebooks/`). For convenience, the majority of the evaluation output is preserved. Auxilary source code is moved to `.py` modules (`src/`). 

### Experiments
- `notebooks/W2GN_toy_experiments.ipynb` -- **toy experiments** (2D: Swiss Roll, 100 Gaussuans, ...);
- `notebooks/W2GN_gaussians_high_dimensions.ipynb` -- optimal maps between **Gaussians in high dimensions**;
- `notebooks/W2GN_latent_space_optimal_transport.ipynb` -- **latent space optimal transport** for a *CelebA 64x64* Aligned Images (use [this script](https://github.com/joeylitalien/celeba-gan-pytorch/blob/master/CelebA_helper.py) to rescale dataset to 64x64);
- `notebooks/W2GN_domain_adaptation.ipynb` -- **domain adaptation** for *MNIST-USPS* digits datasets;
- `notebooks/W2GN_color_transfer.ipynb` -- cycle monotone pixel-wise image-to-image **color transfer** (example images are provided in `data/color_transfer/`);
- `notebooks/W2GN_style_transfer.ipynb` -- cycle monotone image dataset-to-dataset **style transfer** (used datasets are publicitly available at the official [CycleGan repo](https://github.com/junyanz/CycleGAN));
### Input convex neural networks
- `src/icnn.py` -- modules for Input Convex Neural Network architectures (**DenseICNN**, **ConvICNN**);
<p align="center"><img src="pics/icnn.png" width="450" /></p>

## Results
### Toy Experiments
Transforming single Gaussian to the mixture of 100 Gaussuans without mode dropping/collapse.
<p align="center"><img src="pics/toy_100g.png" width="650"/></p>

### Latent Space Optimal Transport
CelebA 64x64 generated faces. The quality of the model highly depends on the quality of the autoencoder. Use `notebooks/AE_Celeba.ipynb` to train MSE or perceptual AE (on VGG features,  to improve AE visual quality).<br>
**Pre-trained autoencoders:** MSE-AE [[Goodle Drive](https://drive.google.com/file/d/17hndo5flmEsGhOP1taWHlAMXMYOUO7pM/view?usp=sharing), [Yandex Disk](https://yadi.sk/d/HRciVy7chhwvAg)], VGG-AE [[Google Drive](https://drive.google.com/file/d/1p1LjGdOw7M3SQ1Zp1BiPKINaKZsOJ3RD/view?usp=sharing), [Yandex Disk](https://yadi.sk/d/BdWCkWuHogTzDQ)].
<p align="center"><img src="pics/latent_ot.png" width="400"/></p>
<p align="center"><img src="pics/celeba_generated_vgg_ae.png" width="700" /></p>

### Image-to-Image Color Transfer
Cycle monotone color transfer is applicable even to gigapixel images!
<p align="center"><img src="pics/colortrans_houses_images.png" width="700" /></p>
<p align="center"><img src="pics/colortrans_houses_palettes.png" width="700" /></p>

### Domain Adaptation
MNIST-USPS domain adaptation. PCA Visualization of feature spaces (see the paper for metrics).
<p align="center"><img src="pics/domain_pca.png" width="700" /></p>

### Unpaired Image-to-Image Style Transfer
Optimal transport map in the space of images. Photo2Cezanne and Winter2Summer datasets are used.
<p align="center"><img src="pics/image_ot.png" width="400" /></p>
<p align="center"><img src="pics/photo_to_cezanne.png" height="214" hspace="25" /><img src="pics/winter_to_summer.png" height="214" /></p>
