import torch
import torch.nn as nn

def freeze_vars(model, var_name, freeze_bn=False):
    """freeze vars. If freeze_bn then only freeze batch_norm params."""
    assert var_name in ["weight", "bias", "alpha", "alpha_bias", "beta"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or not isinstance(v, (nn.GroupNorm, nn.GroupNorm)) or freeze_bn:
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = False

def unfreeze_vars(model, var_name):
    """unfreeze vars."""
    assert var_name in ["weight", "bias", "alpha", "alpha_bias", 'beta']
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if getattr(v, var_name) is not None:
                getattr(v, var_name).requires_grad = True
                
def switch_to_ood(model):
    unfreeze_vars(model, "alpha")
    unfreeze_vars(model, "alpha_bias")
    unfreeze_vars(model, "beta")
    freeze_vars(model, "weight")
    freeze_vars(model, "bias")

def switch_to_finetune(model):
    freeze_vars(model, "alpha")
    freeze_vars(model, "alpha_bias")
    freeze_vars(model, "beta")
    unfreeze_vars(model, "weight")
    unfreeze_vars(model, "bias")

def switch_to_bilevel(model):
    unfreeze_vars(model, "alpha")
    unfreeze_vars(model, "alpha_bias")
    unfreeze_vars(model, "beta")
    unfreeze_vars(model, "weight")
    unfreeze_vars(model, "bias")
 
def freeze_vars_frozen(model, var_name, module_name=None, freeze_bn=False):
    """freeze vars. If freeze_bn then only freeze batch_norm params."""
    assert var_name in ["weight", "bias"]
    if module_name == None:    
        for i, v in model.named_modules():
            if hasattr(v, var_name):
                if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or not isinstance(v, (nn.GroupNorm, nn.GroupNorm)) or freeze_bn:
                    if getattr(v, var_name) is not None:
                        getattr(v, var_name).requires_grad = False
    else:
        if module_name == 'conv1':
            for i, v in model[0].conv1.named_modules():
                if hasattr(v, var_name):
                    if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or not isinstance(v, (nn.GroupNorm, nn.GroupNorm)) or freeze_bn:
                        if getattr(v, var_name) is not None:
                            getattr(v, var_name).requires_grad = False
        elif module_name == 'conv2':
            for i, v in model[0].conv2.named_modules():
                if hasattr(v, var_name):
                    if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or not isinstance(v, (nn.GroupNorm, nn.GroupNorm)) or freeze_bn:
                        if getattr(v, var_name) is not None:
                            getattr(v, var_name).requires_grad = False
        elif module_name == 'conv3':
            for i, v in model[0].conv3.named_modules():
                if hasattr(v, var_name):
                    if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or not isinstance(v, (nn.GroupNorm, nn.GroupNorm)) or freeze_bn:
                        if getattr(v, var_name) is not None:
                            getattr(v, var_name).requires_grad = False
        elif module_name == 'conv4':
            for i, v in model[0].conv4.named_modules():
                if hasattr(v, var_name):
                    if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or not isinstance(v, (nn.GroupNorm, nn.GroupNorm)) or freeze_bn:
                        if getattr(v, var_name) is not None:
                            getattr(v, var_name).requires_grad = False
                            
        for i, v in model[1].named_modules():
            if hasattr(v, var_name):
                if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or not isinstance(v, (nn.GroupNorm, nn.GroupNorm)) or freeze_bn:
                    if getattr(v, var_name) is not None:
                        getattr(v, var_name).requires_grad = False
                        
                        
def unfreeze_vars_frozen(model, var_name, moduel_name=None):
    """unfreeze vars."""
    assert var_name in ["weight", "bias"]
    if moduel_name == None:
        for i, v in model.named_modules():
            if hasattr(v, var_name):
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = True
    else:
        if moduel_name == "conv1":
            for i, v in model[0].conv1.named_modules():
                if hasattr(v, var_name):
                    if getattr(v, var_name) is not None:
                        getattr(v, var_name).requires_grad = True
        elif moduel_name == "conv2":
            for i, v in model[0].conv2.named_modules():
                if hasattr(v, var_name):
                    if getattr(v, var_name) is not None:
                        getattr(v, var_name).requires_grad = True
        elif moduel_name == "conv3":
            for i, v in model[0].conv3.named_modules():
                if hasattr(v, var_name):
                    if getattr(v, var_name) is not None:
                        getattr(v, var_name).requires_grad = True
        elif moduel_name == "conv4":
            for i, v in model[0].conv4.named_modules():
                if hasattr(v, var_name):
                    if getattr(v, var_name) is not None:
                        getattr(v, var_name).requires_grad = True
                         
        for i, v in model[1].named_modules():
            if hasattr(v, var_name):
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = True
            
               
def switch_to_ood_frozen(model, hparams):
    unfreeze_vars_frozen(model, "weight", hparams['frozen'])
    unfreeze_vars_frozen(model, "bias", hparams['frozen'])
    if hparams['frozen'] == 'conv1':
        freeze_vars_frozen(model, "weight", 'conv2')
        freeze_vars_frozen(model, "bias", 'conv2')
        freeze_vars_frozen(model, "weight", 'conv3')
        freeze_vars_frozen(model, "bias", 'conv3')
        freeze_vars_frozen(model, "weight", 'conv4')
        freeze_vars_frozen(model, "bias", 'conv4')
    elif hparams['frozen'] == 'conv2':
        freeze_vars_frozen(model, "weight", 'conv1')
        freeze_vars_frozen(model, "bias", 'conv1')
        freeze_vars_frozen(model, "weight", 'conv3')
        freeze_vars_frozen(model, "bias", 'conv3')
        freeze_vars_frozen(model, "weight", 'conv4')
        freeze_vars_frozen(model, "bias", 'conv4')
    elif hparams['frozen'] == 'conv3':
        freeze_vars_frozen(model, "weight", 'conv1')
        freeze_vars_frozen(model, "bias", 'conv1')
        freeze_vars_frozen(model, "weight", 'conv2')
        freeze_vars_frozen(model, "bias", 'conv2')
        freeze_vars_frozen(model, "weight", 'conv4')
        freeze_vars_frozen(model, "bias", 'conv4')
    elif hparams['frozen'] == 'conv4':
        freeze_vars_frozen(model, "weight", 'conv1')
        freeze_vars_frozen(model, "bias", 'conv1')
        freeze_vars_frozen(model, "weight", 'conv2')
        freeze_vars_frozen(model, "bias", 'conv2')
        freeze_vars_frozen(model, "weight", 'conv3')
        freeze_vars_frozen(model, "bias", 'conv3')
        
def switch_to_finetune_frozen(model, hparams):
    freeze_vars_frozen(model, "weight", hparams['frozen'])
    freeze_vars_frozen(model, "bias", hparams['frozen'])
    if hparams['frozen'] == 'conv1':
        unfreeze_vars_frozen(model, "weight", 'conv2')
        unfreeze_vars_frozen(model, "bias", 'conv2')
        unfreeze_vars_frozen(model, "weight", 'conv3')
        unfreeze_vars_frozen(model, "bias", 'conv3')
        unfreeze_vars_frozen(model, "weight", 'conv4')
        unfreeze_vars_frozen(model, "bias", 'conv4')
    elif hparams['frozen'] == 'conv2':
        unfreeze_vars_frozen(model, "weight", 'conv1')
        unfreeze_vars_frozen(model, "bias", 'conv1')
        unfreeze_vars_frozen(model, "weight", 'conv3')
        unfreeze_vars_frozen(model, "bias", 'conv3')
        unfreeze_vars_frozen(model, "weight", 'conv4')
        unfreeze_vars_frozen(model, "bias", 'conv4')
    elif hparams['frozen'] == 'conv3':
        unfreeze_vars_frozen(model, "weight", 'conv1')
        unfreeze_vars_frozen(model, "bias", 'conv1')
        unfreeze_vars_frozen(model, "weight", 'conv2')
        unfreeze_vars_frozen(model, "bias", 'conv2')
        unfreeze_vars_frozen(model, "weight", 'conv4')
        unfreeze_vars_frozen(model, "bias", 'conv4')
    elif hparams['frozen'] == 'conv4':
        unfreeze_vars_frozen(model, "weight", 'conv1')
        unfreeze_vars_frozen(model, "bias", 'conv1')
        unfreeze_vars_frozen(model, "weight", 'conv2')
        unfreeze_vars_frozen(model, "bias", 'conv2')
        unfreeze_vars_frozen(model, "weight", 'conv3')
        unfreeze_vars_frozen(model, "bias", 'conv3')
    
def switch_to_bilevel_frozen(model):
    unfreeze_vars_frozen(model, "weight")
    unfreeze_vars_frozen(model, "bias")
    

def reset_model(model, map_init, map_ad_type):
    # initialize adapters (alpha)
    for k, v in model.named_parameters():
        if 'alpha' in k:
            # initialize each adapter as an identity matrix
            if map_init == 'eye':
                if v.size(0) > 1:
                    v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)
                else:
                    v.data = torch.ones(v.size()).to(v.device)
                # for residual adapter, each adapter is initialized as identity matrix scaled by 0.0001
                if  map_ad_type == 'residual':
                    v.data = v.data * 0.0001
                if 'bias' in k:
                    v.data = v.data * 0
            elif map_init == 'random':
                # randomly initialization
                v.data = torch.rand(v.data.size()).data.normal_(0, 0.001).to(v.device)
        # initialize pre-classifier alignment mapping (beta)
        if 'beta' in k:
            v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)
    
