import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvexQuadratic
    
class DenseICNN(nn.Module):
    '''Fully Conncted ICNN with input-quadratic skip connections'''
    def __init__(
        self, in_dim, 
        hidden_layer_sizes=[32, 32, 32],
        rank=1, activation='celu', dropout=0.03,
        strong_convexity=1e-6, auto_convexify=True
    ):
        super(DenseICNN, self).__init__()
        
        self.strong_convexity = strong_convexity
        self.hidden_layer_sizes = hidden_layer_sizes
        self.droput = dropout
        self.activation = activation
        self.rank = rank
        self.auto_convexify = auto_convexify
        
        self.quadratic_layers = nn.ModuleList([
            nn.Sequential(
                ConvexQuadratic(in_dim, out_features, rank=rank, bias=True),
                nn.Dropout(dropout)
            )
            for out_features in hidden_layer_sizes
        ])
        
        sizes = zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:])
        self.convex_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
                nn.Dropout(dropout)
            )
            for (in_features, out_features) in sizes
        ])
        
        self.final_layer = nn.Linear(hidden_layer_sizes[-1], 1, bias=False)

    def forward(self, input):
        output = self.quadratic_layers[0](input)
        for quadratic_layer, convex_layer in zip(self.quadratic_layers[1:], self.convex_layers):
            output = convex_layer(output) + quadratic_layer(input)
            if self.activation == 'celu':
                output = torch.celu(output)
            elif self.activation == 'softplus':
                output = F.softplus(output)
            else:
                raise Exception('Activation is not specified or unknown.')
        
        return self.final_layer(output) + .5 * self.strong_convexity * (input ** 2).sum(dim=1).reshape(-1, 1)
    
    def push(self, input):
        output = autograd.grad(
            outputs=self.forward(input), inputs=input,
            create_graph=True, retain_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((input.size()[0], 1)).cuda().float()
        )[0]
        return output    
    
    def convexify(self):
        for layer in self.convex_layers:
            for sublayer in layer:
                if (isinstance(sublayer, nn.Linear)):
                    sublayer.weight.data.clamp_(0)
        self.final_layer.weight.data.clamp_(0)
        

        
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(-1, *self.shape)

class ConvICNN128(nn.Module):
    def __init__(self, channels=3):
        super(ConvICNN128, self).__init__()

        self.first_linear = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
        )
        
        self.first_squared = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
        )
        
        self.convex = nn.Sequential(
            nn.CELU(),
            nn.Conv2d(128, 128, kernel_size=3,stride=2, bias=True, padding=1),  
            nn.CELU(),
            nn.Conv2d(128, 128, kernel_size=3,stride=2, bias=True, padding=1), 
            nn.CELU(),
            nn.Conv2d(128, 128, kernel_size=3,stride=2, bias=True, padding=1), 
            nn.CELU(),
            nn.Conv2d(128, 128, kernel_size=3,stride=2, bias=True, padding=1), 
            nn.CELU(),
            nn.Conv2d(128, 128, kernel_size=3,stride=2, bias=True, padding=1), 
            nn.CELU(),
            View(32* 8 * 8),
            nn.CELU(), 
            nn.Linear(32 * 8 * 8, 128),
            nn.CELU(), 
            nn.Linear(128, 64),
            nn.CELU(), 
            nn.Linear(64, 32),
            nn.CELU(), 
            nn.Linear(32, 1),
            View()
        ).cuda()

    def forward(self, input): 
        output = self.first_linear(input) + self.first_squared(input) ** 2
        output = self.convex(output)
        return output
    
    def push(self, input):
        return autograd.grad(
            outputs=self.forward(input), inputs=input,
            create_graph=True, retain_graph=True,
            only_inputs=True, grad_outputs=torch.ones(input.size()[0]).cuda().float()
        )[0]
    
    def convexify(self):
        for layer in self.convex:
            if (isinstance(layer, nn.Linear)) or (isinstance(layer, nn.Conv2d)):
                layer.weight.data.clamp_(0)