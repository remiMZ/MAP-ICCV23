# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F

from domainbed.models import wide_resnet
from domainbed.models import addition_network
from domainbed.models import coco_resnet 
from domainbed.models import adapter

def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x

class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=hparams['pretrained'])
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=hparams['pretrained'])
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn_ = hparams['freeze_resnet_bn']
        if self.freeze_bn_:
            self.freeze_bn()
        
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        out = self.dropout(self.network(x))
        if 'map_opt' in self.hparams:
            if 'beta' in self.hparams['map_opt']:
                out = self.beta(out)
        return out

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_bn_:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                       
class MNIST_MLP(nn.Module):
    def __init__(self, input_shape, hdim=390):
        super(MNIST_MLP, self).__init__()
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.modules_ = nn.Sequential(
            nn.Linear(input_dim, hdim),
            nn.ReLU(True),
            nn.Linear(hdim, hdim),
            nn.ReLU(True)
        )
        self.n_outputs = hdim
        
        for m in self.modules_:
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain('relu')
                nn.init.xavier_uniform_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.modules_(x)

class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128
    def __init__(self, input_shape, hparams):
        super(MNIST_CNN, self).__init__()
        self.hparams = hparams
        self.is_BN = self.hparams['is_BN'] if 'is_BN' in self.hparams else True
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)
    
        if self.is_BN:
            self.bn0 = nn.GroupNorm(8, 64)
            self.bn1 = nn.GroupNorm(8, 128)
            self.bn2 = nn.GroupNorm(8, 128)
            self.bn3 = nn.GroupNorm(8, 128)
        
        # if is_BN:
        #     self.bn0 = nn.BatchNorm2d(64)
        #     self.bn1 = nn.BatchNorm2d(128)
        #     self.bn2 = nn.BatchNorm2d(128)
        #     self.bn3 = nn.BatchNorm2d(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        if self.is_BN:
            x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        if self.is_BN:
            x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        if self.is_BN:
            x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        if self.is_BN:
            x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        if 'map_opt' in self.hparams:
            if 'beta' in self.hparams['map_opt']:
                x = self.beta(x)
        return x

class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)

def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if "addition_network" in hparams and hparams["addition_network"] == True:
        if "used_map" in hparams and hparams["used_map"] == True:
            addition_network.FeatureWiseTransformation2d_fw.feature_augment = True
            addition_network.ConvBlock.maml = True
            addition_network.SimpleBlock.maml = True
            addition_network.ResNet.maml = True
            network = addition_network.model_dict[hparams["backbone"]]
        else:
            network = addition_network.model_dict[hparams["backbone"]]
        return network()
    elif "used_map" in hparams and hparams["used_map"] == True:
        if input_shape[1:3] == (28, 28):
            orig_model = MNIST_CNN(input_shape, hparams)
            return adapter.cnn_map(orig_model, hparams)
        elif input_shape[1:3] == (64, 64):
            orig_model = coco_resnet.ResNet8(input_shape, hparams)
            return adapter.resnet_map(orig_model, hparams)
        elif input_shape[1:3] == (224, 224):
            orig_model = ResNet(input_shape, hparams)
            return adapter.resnet_map(orig_model, hparams)
        else:
            raise NotImplementedError
    else:
        if len(input_shape) == 1:
            return MLP(input_shape[0], hparams["mlp_width"], hparams)
        elif input_shape[1:3] == (14, 14):  # ColoredMNIST_IRM
            return MNIST_MLP(input_shape)
        elif input_shape[1:3] == (28, 28):
            return MNIST_CNN(input_shape, hparams)
        elif input_shape[1:3] == (32, 32):
            return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
        elif input_shape[1:3] == (64, 64):
            return coco_resnet.ResNet8(input_shape, hparams)
        elif input_shape[1:3] == (224, 224):
            return ResNet(input_shape, hparams)
        else:
            raise NotImplementedError

def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)

class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd, reverse=True):
        ctx.lambd = lambd
        ctx.reverse=reverse
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reverse:
            return (grad_output * -ctx.lambd), None, None
        else:
            return (grad_output * ctx.lambd), None, None

class DisNet(nn.Module):
    def __init__(self, in_channels, num_domains, layers=[1024, 256]):
        super(DisNet, self).__init__()
        self.domain_classifier = nn.ModuleList()
        # different from the original implementation, a single linear layer is
        # used for fair comparison with other algorithms
        self.domain_classifier = nn.Linear(in_channels, num_domains)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, .1)
                nn.init.constant_(m.bias, 0.)

        self.lambda_ = 0.0

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

    def forward(self, x):
        x = GradReverse.apply(x, self.lambda_, True)
        return self.domain_classifier(x)

    def get_params(self, lr):
        return [{'params': self.domain_classifier.parameters(), 'lr': lr}]

class ClsNet(nn.Module):
    def __init__(self, in_channels, num_domains, num_classes, reverse=True,
                 layers=[1024, 256]):
        super(ClsNet, self).__init__()
        self.classifier_list = nn.ModuleList()
        for _ in range(num_domains):
            # different from the original implementation, a single linear layer is
            # used for fair comparison with other algorithms
            self.classifier_list.append(nn.Linear(in_channels, num_classes))
        for m in self.classifier_list.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, .1)
                nn.init.constant_(m.bias, 0.)

        self.lambda_ = 0
        self.reverse = reverse

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

    def forward(self, x):
        output = []
        for c, x_ in zip(self.classifier_list, x):
            if len(x_) == 0:
                output.append(None)
            else:
                x_ = GradReverse.apply(x_, self.lambda_, self.reverse)
                output.append(c(x_))

        return output

    def get_params(self, lr):
        return [{'params': self.classifier_list.parameters(), 'lr': lr}]
    
if __name__ =="__main__":
    backbone = MNIST_CNN(20, True)
    print(backbone)