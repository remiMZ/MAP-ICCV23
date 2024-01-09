import copy
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class conv_map(nn.Module):
    def __init__(self, orig_conv, hparams):
        super(conv_map, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        planes, in_planes, _, _ = self.conv.weight.size()
        stride, _ = self.conv.stride
        if 'alpha' not in hparams['map_opt']:
            self.ad_type = 'none'
        else:
            self.ad_type = hparams['map_ad_type']
            self.ad_form = hparams['map_ad_form']
        
        if self.ad_type == 'residual':
            if self.ad_form == 'matrix' or planes != in_planes:
                self.alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
            else:
                self.alpha = nn.Parameter(torch.ones(1, planes, 1,1))
        elif self.ad_type == 'serial':
            if self.ad_form == 'matrix':
                self.alpha = nn.Parameter(torch.ones(planes, planes, 1, 1))
            else:
                self.alpha = nn.Parameter(torch.ones(1, planes, 1, 1))
                
            self.alpha_bias = nn.Parameter(torch.ones(1, planes, 1, 1))
            self.alpha_bias.requires_grad = True
        if self.ad_type != 'none':
            self.alpha.requires_grad = True

    def forward(self, x):
        y = self.conv(x)
        if self.ad_type == 'residual':
            if self.alpha.size(0) > 1:
                # residual adaptation in matrix form
                y = y + F.conv2d(x, self.alpha, stride=self.conv.stride)
            else:
                # residual adaptation in channel-wise (vector)
                y = y + x * self.alpha
        elif self.ad_type == 'serial':
            if self.alpha.size(0) > 1:
                # serial adaptation in matrix form
                y = F.conv2d(y, self.alpha) + self.alpha_bias
            else:
                # serial adaptation in channel-wise (vector)
                y = y * self.alpha + self.alpha_bias
        return y
    
class pa(nn.Module):
    """ 
    pre-classifier alignment (PA) mapping from 'Universal Representation Learning from Multiple Domains for Few-shot Classification'
    (https://arxiv.org/pdf/2103.13841.pdf)
    """
    def __init__(self, feat_dim):
        super(pa, self).__init__()
        # define pre-classifier alignment mapping
        self.weight = nn.Parameter(torch.ones(feat_dim, feat_dim, 1, 1))
        self.weight.requires_grad = True
        
    def forward(self, x):
        if len(list(x.size())) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.conv2d(x, self.weight.to(x.device)).flatten(1)
        return x
   
class cnn_map(nn.Module):
    """Attaching adapter to the CNN backbone"""
    def __init__(self, orig_cnn, hparams):
        super(cnn_map, self).__init__()
        for name, m in orig_cnn.named_children():
            if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                new_conv = conv_map(m, hparams)
                setattr(orig_cnn, name, new_conv)
        self.backbone = orig_cnn
        self.n_outputs = self.backbone.n_outputs
        
        if 'beta' in hparams['map_opt']:
            # attach per-classifier alignment mapping (beta)
            beta = pa(self.n_outputs)
            setattr(self.backbone, 'beta', beta)
                
    def forward(self, x):
        return self.backbone.forward(x)

class resnet_map(nn.Module):
    """Attaching adapter to the ResNet backbone"""
    def __init__(self, orig_resnet, hparams):
        super(resnet_map, self).__init__()
        self.hparams = hparams
        # attaching adapter to each convolutional layers and note that we only attach adapter to residual blocks in the ResNet
        if hparams['resnet18']:
             orig_res = orig_resnet.network
        else:
            orig_res = orig_resnet
            
        for block in orig_res.layer1:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_map(m, hparams)
                    setattr(block, name, new_conv)

        for block in orig_res.layer2:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_map(m, hparams)
                    setattr(block, name, new_conv)

        for block in orig_res.layer3:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_map(m, hparams)
                    setattr(block, name, new_conv)

        if hparams['resnet18']:
            for block in orig_res.layer4:
                for name, m in block.named_children():
                    if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                        new_conv = conv_map(m, hparams)
                        setattr(block, name, new_conv)

        self.backbone = orig_resnet
        self.n_outputs = orig_resnet.n_outputs

        if 'beta' in hparams['map_opt']:
            # attach per-classifier alignment mapping (beta)
            beta = pa(self.n_outputs)
            setattr(self.backbone, 'beta', beta)
        
    def forward(self, x):
        return self.backbone.forward(x)
        
    def train(self, mode=True):
        self.backbone.train(mode)

    def freeze_bn(self):
        self.backbone.freeze_bn
    