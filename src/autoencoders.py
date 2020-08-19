import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models 

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class CelebaEncoder(nn.Module):
    """ Celeba Encoder
        Args:
            init_num_filters (int): initial number of filters from encoder image channels
            lrelu_slope (float): positive number indicating LeakyReLU negative slope
            inter_fc_dim (int): intermediate fully connected dimensionality prior to embedding layer
            embedding_dim (int): embedding dimensionality
    """
    def __init__(self, init_num_filters=16, lrelu_slope=0.2, inter_fc_dim=128, embedding_dim=2, nc=3, dropout=0.05):
        super(CelebaEncoder, self).__init__()

        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.inter_fc_dim_ = inter_fc_dim
        self.embedding_dim_ = embedding_dim

        self.features = nn.Sequential(
            nn.Conv2d(nc,  self.init_num_filters_ * 1, 4, 2, 1, bias=False),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            nn.Dropout(dropout),
            
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.init_num_filters_, self.init_num_filters_ * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 2),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            nn.Dropout(dropout),
            
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.init_num_filters_  * 2, self.init_num_filters_ * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_  * 4),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            nn.Dropout(dropout),
            
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.init_num_filters_  * 4, self.init_num_filters_ * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_  * 8),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            nn.Dropout(dropout),
            
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.init_num_filters_ * 8, self.init_num_filters_ * 8, 4, 2, 0, bias=False),
        )
        
        self.fc_out = nn.Sequential(
            nn.Linear(self.init_num_filters_ * 8, self.embedding_dim_),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            nn.BatchNorm1d(self.embedding_dim_, affine=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class CelebaDecoder(nn.Module):
    """ Celeba Decoder
        Args:
            init_num_filters (int): initial number of filters from encoder image channels
            lrelu_slope (float): positive number indicating LeakyReLU negative slope
            inter_fc_dim (int): intermediate fully connected dimensionality prior to embedding layer
            embedding_dim (int): embedding dimensionality
    """
    def __init__(self, init_num_filters=16, lrelu_slope=0.2, inter_fc_dim=128, embedding_dim=2, nc=3, dropout=0.05):
        super(CelebaDecoder, self).__init__()

        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.inter_fc_dim_ = inter_fc_dim
        self.embedding_dim_ = embedding_dim

        self.features = nn.Sequential(
            nn.ConvTranspose2d(self.init_num_filters_ * 8, self.init_num_filters_ * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 8),
            nn.ReLU(True),
            nn.Dropout(dropout),
            
            nn.ConvTranspose2d(self.init_num_filters_ * 8, self.init_num_filters_ * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 4),
            nn.ReLU(True),
            nn.Dropout(dropout),

            nn.ConvTranspose2d(self.init_num_filters_ * 4, self.init_num_filters_ * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            
            nn.ConvTranspose2d(self.init_num_filters_ * 2, self.init_num_filters_ * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.init_num_filters_ * 1),
            nn.ReLU(True),
            nn.Dropout(dropout),
            
            nn.ConvTranspose2d(self.init_num_filters_ * 1, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        self.fc_in = nn.Sequential(
            nn.Linear(self.embedding_dim_, self.init_num_filters_ * 8),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            nn.BatchNorm1d(self.init_num_filters_ * 8)
        )

    def forward(self, z):
        z = self.fc_in(z)
        z = z.view(-1, self.init_num_filters_ * 8, 1, 1)
        z = self.features(z)
        return z
    
class CelebaAutoencoder(nn.Module):
    """ Celeba Autoencoder
        Args:
            init_num_filters (int): initial number of filters from encoder image channels
            lrelu_slope (float): positive number indicating LeakyReLU negative slope
            inter_fc_dim (int): intermediate fully connected dimensionality prior to embedding layer
            embedding_dim (int): embedding dimensionality
    """
    def __init__(self, init_num_filters=16, lrelu_slope=0.2, inter_fc_dim=128, embedding_dim=2):
        super(CelebaAutoencoder, self).__init__()

        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.inter_fc_dim_ = inter_fc_dim
        self.embedding_dim_ = embedding_dim

        self.encoder = CelebaEncoder(init_num_filters, lrelu_slope, inter_fc_dim, embedding_dim)
        self.decoder = CelebaDecoder(init_num_filters, lrelu_slope, inter_fc_dim, embedding_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
    
class VGGPerceptualLoss(nn.Module):
    def __init__(self, vgg=None, layer_names=['3', '8', '15', '22']):
        super(VGGPerceptualLoss, self).__init__()
        if vgg is None:
            self.vgg_layers = models.vgg16(pretrained=True).features
        else:
            self.vgg = vgg.features
        self.layer_names = layer_names
        self.eval()
        
    def __call__(self, in_1, in_2):
        assert in_1.shape == in_2.shape
        loss = 0.
        out_1 = in_1; out_2 = in_2
        for name, module in self.vgg_layers._modules.items():
            out_1 = module(out_1); out_2 = module(out_2);
            if name in self.layer_names:
                loss += F.mse_loss(out_1, out_2)
        return loss