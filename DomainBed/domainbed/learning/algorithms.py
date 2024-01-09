# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np
from collections import OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from domainbed.models import networks
from domainbed.utils.misc import (
    random_pairs_of_minibatches, 
    ParamDict, 
    MovingAverage, 
    l2_between_dicts, 
    proj,
)
from domainbed.utils.map_models import (
    switch_to_bilevel,
    switch_to_ood,
    switch_to_finetune, 
    reset_model,
    switch_to_bilevel_frozen,
    switch_to_ood_frozen,
    switch_to_finetune_frozen
)

# all algorithms = 30
ALGORITHMS = [
    'ERM',
    'IRM',
    'VREx',
    'ARM',
    'GroupDRO',
    'MLDG',
    'MMD',
    'IGA',
    'SANDMask',
    'Fish',
    'CDANN',
    'TRM',
    'IB_ERM',
    'IB_IRM',
    'CondCAD',
    'CausIRL_CORAL', 
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
    
#! ERM
class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        
        # load network
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
    
        self.network = nn.Sequential(self.featurizer, self.classifier)
                
        # optimizer
        self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        
        # optimizer for bi-level
        if "used_map" in self.hparams and self.hparams["used_map"] == True:
            reset_model(self.network, self.hparams['map_init'], self.hparams["map_ad_type"])
            alpha_params = [v for k, v in self.network.named_parameters() if 'alpha' in k]
            beta_params = [v for k, v in self.network.named_parameters() if 'beta' in k]
            map_params = []
            if 'alpha' in self.hparams['map_opt']:
                map_params.append({'params': alpha_params, 'lr': self.hparams['lr_alpha']})
            if 'beta' in self.hparams['map_opt']:
                map_params.append({'params': beta_params, 'lr': self.hparams["lr_beta"]})
            self.map_optimizer = torch.optim.Adadelta(map_params)
            

    def update(self, minibatches, unlabeled=None, return_loss=False):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        
        if return_loss:
            return 
        else:
            self.optimizer.step()
            return {'loss': loss.item()}
    
    def predict(self, x):
        return self.network(x)
  
#! IRM
class IRM(ERM):
    """Invariant Risk Minimization"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        
    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        
        return result

    def update(self, minibatches, unlabeled=None, return_loss=False):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x,y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        
        if return_loss:
            return 
        else:
            self.optimizer.step()
            self.update_count += 1
            return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}
    
    def update_map(self, minibatches, map_minibatches, unlabeled=None):
        train_x = torch.cat([x for x, y in minibatches])
        train_y = torch.cat([y for x, y in minibatches])
        val_x = torch.cat([x for x, y in map_minibatches])
        val_y = torch.cat([y for x, y in map_minibatches])
        #! 1
        for step in range(self.hparams['adapter_steps']):
            switch_to_ood(self.network)
            train_logits = self.network(train_x)
            loss_map = F.cross_entropy(train_logits, train_y)
            self.map_optimizer.zero_grad()
            loss_map.backward()
            self.map_optimizer.step()
        #! 2
        switch_to_finetune(self.network) 
        step_vals = self.update(minibatches)
        #! 3
        switch_to_bilevel(self.network) 
        self.update(minibatches, return_loss=True)
    
        def grad2vec(parameters):
            grad_vec = []
            for param in parameters:
                grad_vec.append(param.grad.view(-1).detach())
            return torch.cat(grad_vec)
    
        param_grad_vec = grad2vec(self.network.parameters())
        #! 4
        switch_to_ood(self.network)
        self.map_optimizer.zero_grad()
        val_logits = self.network(val_x)
        loss_map = F.cross_entropy(val_logits, val_y)
        loss_map.backward()
        map_grad_vec = grad2vec(self.network.parameters())
        implicit_gradient = - self.hparams['lr2'] * map_grad_vec * param_grad_vec
        
        def append_grad_to_vec(vec, parameters):
            if not isinstance(vec, torch.Tensor):
                raise TypeError('expected torch.Tensor, but got: {}'
                                .format(torch.typename(vec)))
            pointer = 0
            for param in parameters:
                num_param = param.numel()
                param.grad.copy_(param.grad + vec[pointer:pointer + num_param].view_as(param).data)
                pointer += num_param

        append_grad_to_vec(implicit_gradient, self.network.parameters())
        self.map_optimizer.step()
        # self.optimizer.step()

        return {'loss': step_vals['loss'], 
                'nll': step_vals['nll'],
                'penalty': step_vals['penalty']}  
        
    def update_pareto(self, minibatches, map_minibatches, unlabeled=None, return_loss=False):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.
        
        train_x = torch.cat([x for x, y in minibatches])
        train_y = torch.cat([y for x, y in minibatches])
        val_x = torch.cat([x for x, y in map_minibatches])
        
        train_logits = self.network(train_x)
        loss_iid = F.cross_entropy(train_logits, train_y)

        val_logits = self.network(val_x)
        val_logits_idx = 0
        for i, (x, y) in enumerate(map_minibatches):
            logits = val_logits[val_logits_idx:val_logits_idx + x.shape[0]]
            val_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(map_minibatches)
        penalty /= len(map_minibatches) 
        loss_ood = nll + (penalty_weight * penalty)
        
        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
        # Reset Adam, because it doesn't like the sharp jump in gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss = loss_iid + loss_ood
        loss.backward()
    
        if return_loss:
            return 
        else:
            self.optimizer.step()
            self.update_count += 1
            
            return {'loss_iid': loss_iid.item(), 'loss_ood': loss_ood.item(), 'nll': nll.item(), 'penalty': penalty.item()}
                        

#! VREx    
class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None, return_loss=False):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        
        if return_loss:
            return 
        else:    
            self.optimizer.step()
            self.update_count += 1
            return {'loss': loss.item(), 'nll': nll.item(),
                    'penalty': penalty.item()}
    
    def update_map(self, minibatches, map_minibatches, unlabeled=None):
        train_x = torch.cat([x for x, y in minibatches])
        train_y = torch.cat([y for x, y in minibatches])
        val_x = torch.cat([x for x, y in map_minibatches])
        val_y = torch.cat([y for x, y in map_minibatches])
        #! 1
        for step in range(self.hparams['adapter_steps']):
            switch_to_ood(self.network)
            train_logits = self.network(train_x)
            loss_map = F.cross_entropy(train_logits, train_y)
            self.map_optimizer.zero_grad()
            loss_map.backward()
            self.map_optimizer.step()
        #! 2
        switch_to_finetune(self.network) 
        step_vals = self.update(minibatches)
        #! 3
        switch_to_bilevel(self.network) 
        self.update(minibatches, return_loss=True)
        
        def grad2vec(parameters):
            grad_vec = []
            for param in parameters:
                grad_vec.append(param.grad.view(-1).detach())
            return torch.cat(grad_vec)
    
        param_grad_vec = grad2vec(self.network.parameters())
        #! 4
        switch_to_ood(self.network)
        self.map_optimizer.zero_grad()
        val_logits = self.network(val_x)
        loss_map = F.cross_entropy(val_logits, val_y)
        loss_map.backward()
        map_grad_vec = grad2vec(self.network.parameters())
        implicit_gradient = - self.hparams['lr2'] * map_grad_vec * param_grad_vec
        
        def append_grad_to_vec(vec, parameters):
            if not isinstance(vec, torch.Tensor):
                raise TypeError('expected torch.Tensor, but got: {}'
                                .format(torch.typename(vec)))
            pointer = 0
            for param in parameters:
                num_param = param.numel()
                param.grad.copy_(param.grad + vec[pointer:pointer + num_param].view_as(param).data)
                pointer += num_param

        append_grad_to_vec(implicit_gradient, self.network.parameters())
        self.map_optimizer.step()
        # self.optimizer.step()
    
        return {'loss': step_vals['loss'], 
                'nll': step_vals['nll'],
                'penalty': step_vals['penalty']}
        
    def update_frozen(self, minibatches, map_minibatches, unlabeled=None):
        train_x = torch.cat([x for x, y in minibatches])
        train_y = torch.cat([y for x, y in minibatches])
        val_x = torch.cat([x for x, y in map_minibatches])
        val_y = torch.cat([y for x, y in map_minibatches])
        #! 1
        for step in range(self.hparams['adapter_steps']):
            switch_to_ood_frozen(self.network, self.hparams) 
            train_logits = self.network(train_x)
            loss_map = F.cross_entropy(train_logits, train_y)
            # self.map_optimizer.zero_grad()
            loss_map.backward()
            self.optimizer.step()
        #! 2
        switch_to_finetune_frozen(self.network, self.hparams) 
        step_vals = self.update(minibatches)
        #! 3
        switch_to_bilevel_frozen(self.network) 
        self.update(minibatches, return_loss=True)
        
        def grad2vec(parameters):
            grad_vec = []
            for param in parameters:
                grad_vec.append(param.grad.view(-1).detach())
            return torch.cat(grad_vec)
    
        param_grad_vec = grad2vec(self.network.parameters())
        #! 4
        switch_to_ood_frozen(self.network, self.hparams)
        self.optimizer.zero_grad()
        val_logits = self.network(val_x)
        loss_map = F.cross_entropy(val_logits, val_y)
        loss_map.backward()
        map_grad_vec = grad2vec(self.network.parameters())
        implicit_gradient = - self.hparams['lr2'] * map_grad_vec * param_grad_vec
        
        def append_grad_to_vec(vec, parameters):
            if not isinstance(vec, torch.Tensor):
                raise TypeError('expected torch.Tensor, but got: {}'
                                .format(torch.typename(vec)))
            pointer = 0
            for param in parameters:
                num_param = param.numel()
                param.grad.copy_(param.grad + vec[pointer:pointer + num_param].view_as(param).data)
                pointer += num_param

        append_grad_to_vec(implicit_gradient, self.network.parameters())
        # self.map_optimizer.step()
        self.optimizer.step()
    
        return {'loss': step_vals['loss'], 
                'nll': step_vals['nll'],
                'penalty': step_vals['penalty']}

#! ARM 
class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams['batch_size']

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)
    
    def update_map(self, minibatches, map_minibatches, unlabeled=None):
        train_x = torch.cat([x for x, y in minibatches])
        train_y = torch.cat([y for x, y in minibatches])
        val_x = torch.cat([x for x, y in map_minibatches])
        val_y = torch.cat([y for x, y in map_minibatches])
        #! 1
        for step in range(self.hparams['adapter_steps']):
            switch_to_ood(self.network)
            switch_to_ood(self.context_net)
            train_logits = self.predict(train_x)
            loss_map = F.cross_entropy(train_logits, train_y)
            self.map_optimizer.zero_grad()
            loss_map.backward()
            self.map_optimizer.step()
        #! 2
        switch_to_finetune(self.network) 
        switch_to_finetune(self.context_net)
        step_vals = self.update(minibatches)
        #! 3
        switch_to_bilevel(self.network) 
        switch_to_bilevel(self.context_net)
        self.update(minibatches, return_loss=True)
    
        def grad2vec(parameters):
            grad_vec = []
            for param in parameters:
                grad_vec.append(param.grad.view(-1).detach())
            return torch.cat(grad_vec)
    
        param_grad_vec = grad2vec(self.network.parameters())
        #! 4
        switch_to_ood(self.network)
        self.map_optimizer.zero_grad()
        val_logits = self.predict(val_x)
        loss_map = F.cross_entropy(val_logits, val_y)
        loss_map.backward()
        map_grad_vec = grad2vec(self.network.parameters())
        implicit_gradient = - self.hparams['lr2'] * map_grad_vec * param_grad_vec
        
        def append_grad_to_vec(vec, parameters):
            if not isinstance(vec, torch.Tensor):
                raise TypeError('expected torch.Tensor, but got: {}'
                                .format(torch.typename(vec)))
            pointer = 0
            for param in parameters:
                num_param = param.numel()
                param.grad.copy_(param.grad + vec[pointer:pointer + num_param].view_as(param).data)
                pointer += num_param

        append_grad_to_vec(implicit_gradient, self.network.parameters())
        self.map_optimizer.step()
        # self.optimizer.step()

        return {'loss': step_vals['loss']}

#! GroupDRO    
class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None, return_loss=False):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        if return_loss:
            return 
        else:
            self.optimizer.step()
            return {'loss': loss.item()}
        
    def update_map(self, minibatches, map_minibatches, unlabeled=None):
        train_x = torch.cat([x for x, y in minibatches])
        train_y = torch.cat([y for x, y in minibatches])
        val_x = torch.cat([x for x, y in map_minibatches])
        val_y = torch.cat([y for x, y in map_minibatches])
        #! 1
        for step in range(self.hparams['adapter_steps']):
            switch_to_ood(self.network)
            train_logits = self.network(train_x)
            loss_map = F.cross_entropy(train_logits, train_y)
            self.map_optimizer.zero_grad()
            loss_map.backward()
            self.map_optimizer.step()
        #! 2                        
        switch_to_finetune(self.network) 
        step_vals = self.update(minibatches)
        #! 3
        switch_to_bilevel(self.network) 
        self.update(minibatches, return_loss=True)
        
        def grad2vec(parameters):
            grad_vec = []
            for param in parameters:
                grad_vec.append(param.grad.view(-1).detach())
            return torch.cat(grad_vec)
    
        param_grad_vec = grad2vec(self.network.parameters())
        #! 4
        switch_to_ood(self.network)
        self.map_optimizer.zero_grad()
        val_logits = self.network(val_x)
        loss_map = F.cross_entropy(val_logits, val_y)
        loss_map.backward()
        map_grad_vec = grad2vec(self.network.parameters())
        implicit_gradient = - self.hparams['lr2'] * map_grad_vec * param_grad_vec
        
        def append_grad_to_vec(vec, parameters):
            if not isinstance(vec, torch.Tensor):
                raise TypeError('expected torch.Tensor, but got: {}'
                                .format(torch.typename(vec)))
            pointer = 0
            for param in parameters:
                num_param = param.numel()
                param.grad.copy_(param.grad + vec[pointer:pointer + num_param].view_as(param).data)
                pointer += num_param

        append_grad_to_vec(implicit_gradient, self.network.parameters())
        self.map_optimizer.step()
        # self.optimizer.step()
        
        return {'loss': step_vals['loss']}
    
#! MLDG   
class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)

    def update(self, minibatches, unlabeled=None, return_loss=False):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        if return_loss:
            return
        else:
            self.optimizer.step()
            return {'loss': objective}
    
    # This commented "update" method back-propagates through the gradients of
    # the inner update, as suggested in the original MAML paper.  However, this
    # is twice as expensive as the uncommented "update" method, which does not
    # compute second-order derivatives, implementing the First-Order MAML
    # method (FOMAML) described in the original MAML paper.

    # def update(self, minibatches, unlabeled=None):
    #     objective = 0
    #     beta = self.hparams["beta"]
    #     inner_iterations = self.hparams["inner_iterations"]

    #     self.optimizer.zero_grad()

    #     with higher.innerloop_ctx(self.network, self.optimizer,
    #         copy_initial_weights=False) as (inner_network, inner_optimizer):

    #         for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
    #             for inner_iteration in range(inner_iterations):
    #                 li = F.cross_entropy(inner_network(xi), yi)
    #                 inner_optimizer.step(li)
    #
    #             objective += F.cross_entropy(self.network(xi), yi)
    #             objective += beta * F.cross_entropy(inner_network(xj), yj)

    #         objective /= len(minibatches)
    #         objective.backward()
    #
    #     self.optimizer.step()
    #
    #     return objective
    
#! AbstractMMD    
class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None, return_loss=False):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        
        if return_loss:
            return 
        else:
            self.optimizer.step()

            if torch.is_tensor(penalty):
                penalty = penalty.item()

            return {'loss': objective.item(), 'penalty': penalty}
    
#! MMD
class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True) 
        
#! CORAL
class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)
   
#! IGA
class IGA(ERM):
    """
    Inter-environmental Gradient Alignment
    From https://arxiv.org/abs/2008.01883v2
    """

    def __init__(self, in_features, num_classes, num_domains, hparams):
        super(IGA, self).__init__(in_features, num_classes, num_domains, hparams)

    def update(self, minibatches, unlabeled=False, return_loss=False):
        total_loss = 0
        grads = []
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            total_loss += env_loss

            env_grad = autograd.grad(env_loss, self.network.parameters(),
                                        create_graph=True)

            grads.append(env_grad)

        mean_loss = total_loss / len(minibatches)
        mean_grad = autograd.grad(mean_loss, self.network.parameters(),
                                        retain_graph=True)

        # compute trace penalty
        penalty_value = 0
        for grad in grads:
            for g, mean_g in zip(grad, mean_grad):
                penalty_value += (g - mean_g).pow(2).sum()

        objective = mean_loss + self.hparams['penalty'] * penalty_value

        self.optimizer.zero_grad()
        objective.backward()

        if return_loss:
            return 
        else:
            self.optimizer.step()

            return {'loss': mean_loss.item(), 'penalty': penalty_value.item()}
    
#! SANDMask
class SANDMask(ERM):
    """
    SAND-mask: An Enhanced Gradient Masking Strategy for the Discovery of Invariances in Domain Generalization
    <https://arxiv.org/abs/2106.02266>
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]
        self.k = hparams["k"]
        betas = (0.9, 0.999)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],
            betas=betas
        )

        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None, return_loss=False):
        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)
            env_grads = autograd.grad(env_loss, self.network.parameters(), retain_graph=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        # gradient masking applied here
        self.mask_grads(param_gradients, self.network.parameters())
        
        if return_loss:
            return 
        else:
            self.optimizer.step()
            self.update_count += 1
            return {'loss': mean_loss}
    
    def mask_grads(self, gradients, params):
        '''
        Here a mask with continuous values in the range [0,1] is formed to control the amount of update for each parameter based on the agreement of gradients coming from different environments.
        '''
        device = gradients[0][0].device
        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            avg_grad = torch.mean(grads, dim=0)
            grad_signs = torch.sign(grads)
            gamma = torch.tensor(1.0).to(device)
            grads_var = grads.var(dim=0)
            grads_var[torch.isnan(grads_var)] = 1e-17
            lam = (gamma * grads_var).pow(-1)
            mask = torch.tanh(self.k * lam * (torch.abs(grad_signs.mean(dim=0)) - self.tau))
            mask = torch.max(mask, torch.zeros_like(mask))
            mask[torch.isnan(mask)] = 1e-17
            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))

#! Fish            
class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None
        
        # optimizer for bi-level
        if "used_map" in self.hparams and self.hparams["used_map"] == True:
            reset_model(self.network, self.hparams['map_init'], self.hparams["map_ad_type"])
            alpha_params = [v for k, v in self.network.named_parameters() if 'alpha' in k]
            beta_params = [v for k, v in self.network.named_parameters() if 'beta' in k]
            map_params = []
            if 'alpha' in self.hparams['map_opt']:
                map_params.append({'params': alpha_params, 'lr': self.hparams['lr_alpha']})
            if 'beta' in self.hparams['map_opt']:
                map_params.append({'params': beta_params, 'lr': self.hparams["lr_beta"]})
            self.map_optimizer = torch.optim.Adadelta(map_params)

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                            weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None, return_loss=False):
        self.create_clone(minibatches[0][0].device)

        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
                meta_weights=self.network.state_dict(),
                inner_weights=self.network_inner.state_dict(),
                lr_meta=self.hparams["meta_lr"]
            )
        
        if return_loss:
            return 
        else:
            self.network.reset_weights(meta_weights)
            return {'loss': loss.item()}
    
    def predict(self, x):
        return self.network(x)

#! AbstractDANN
class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs,
            num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
            self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
                list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
                list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))
        
        # optimizer for bi-level
        if "used_map" in self.hparams and self.hparams["used_map"] == True:
            reset_model(self.featurizer, self.hparams['map_init'], self.hparams["map_ad_type"])
            alpha_params = [v for k, v in self.featurizer.named_parameters() if 'alpha' in k]
            beta_params = [v for k, v in self.featurizer.named_parameters() if 'beta' in k]
            map_params = []
            if 'alpha' in self.hparams['map_opt']:
                map_params.append({'params': alpha_params, 'lr': self.hparams['lr_alpha']})
            if 'beta' in self.hparams['map_opt']:
                map_params.append({'params': beta_params, 'lr': self.hparams["lr_beta"]})
            self.map_optimizer = torch.optim.Adadelta(map_params)

    def update(self, minibatches, unlabeled=None, return_loss=False):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):
            self.disc_opt.zero_grad()
            disc_loss.backward()
            if return_loss:
                return 
            else:
                self.disc_opt.step()
                return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            if return_loss:
                return 
            else:
                self.gen_opt.step()
                return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))
     
#! CDANN 
class CDANN(AbstractDANN):
    """Conditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=True, class_balance=True)

    def update_map(self, minibatches, map_minibatches, unlabeled=None):
        train_x = torch.cat([x for x, y in minibatches])
        train_y = torch.cat([y for x, y in minibatches])
        val_x = torch.cat([x for x, y in map_minibatches])
        val_y = torch.cat([y for x, y in map_minibatches])
        #! 1 
        for step in range(self.hparams['adapter_steps']):
            switch_to_ood(self.featurizer)
            switch_to_ood(self.classifier)
            switch_to_ood(self.discriminator)
            switch_to_ood(self.class_embeddings)
            train_logits = self.classifier(self.featurizer(train_x))
            loss_map = F.cross_entropy(train_logits, train_y)
            self.map_optimizer.zero_grad()
            loss_map.backward()
            self.map_optimizer.step()
        #! 2
        switch_to_finetune(self.featurizer)
        switch_to_finetune(self.classifier)
        switch_to_finetune(self.discriminator)
        switch_to_finetune(self.class_embeddings)
        step_vals = self.update(minibatches)
        #! 3
        switch_to_bilevel(self.featurizer)
        switch_to_bilevel(self.classifier)
        switch_to_bilevel(self.discriminator)
        switch_to_bilevel(self.class_embeddings)
        self.update(minibatches, return_loss=True)
       
        def grad2vec(parameters):
            grad_vec = []
            for param in parameters:
                grad_vec.append(param.grad.view(-1).detach())
            return torch.cat(grad_vec)
    
        param_grad_vec = grad2vec(self.featurizer.parameters())
        #! 4
        switch_to_ood(self.featurizer)
        switch_to_ood(self.classifier)
        switch_to_ood(self.discriminator)
        switch_to_ood(self.class_embeddings)
        self.map_optimizer.zero_grad()
        val_logits = self.classifier(self.featurizer(val_x))
        loss_map = F.cross_entropy(val_logits, val_y)
        loss_map.backward()
        map_grad_vec = grad2vec(self.featurizer.parameters())
        implicit_gradient = - self.hparams['lr2'] * map_grad_vec * param_grad_vec
        
        def append_grad_to_vec(vec, parameters):
            if not isinstance(vec, torch.Tensor):
                raise TypeError('expected torch.Tensor, but got: {}'
                                .format(torch.typename(vec)))
            pointer = 0
            for param in parameters:
                num_param = param.numel()
                param.grad.copy_(param.grad + vec[pointer:pointer + num_param].view_as(param).data)
                pointer += num_param

        append_grad_to_vec(implicit_gradient, self.featurizer.parameters())
        self.map_optimizer.step()
        # self.optimizer.step()

        if 'disc_loss' in step_vals:
            return {'disc_loss': step_vals['disc_loss']}
        elif 'gen_loss' in step_vals:
            return {'gen_loss': step_vals['gen_loss']}
        
#! TRM
class TRM(Algorithm):
    """
    Learning Representations that Support Robust Transfer of Predictors
    <https://arxiv.org/abs/2110.09940>
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(TRM, self).__init__(input_shape, num_classes, num_domains,hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.num_domains = num_domains
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes).cuda()
        self.clist = [nn.Linear(self.featurizer.n_outputs, num_classes).cuda() for i in range(num_domains+1)]
        self.olist = [torch.optim.SGD(
            self.clist[i].parameters(),
            lr=1e-1,
        ) for i in range(num_domains+1)]

        self.optimizer_f = torch.optim.Adam(
            self.featurizer.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_c = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # initial weights
        self.alpha = torch.ones((num_domains, num_domains)).cuda() - torch.eye(num_domains).cuda()
        
        # optimizer for bi-level
        if "used_map" in self.hparams and self.hparams["used_map"] == True:
            reset_model(self.featurizer, self.hparams['map_init'], self.hparams["map_ad_type"])
            alpha_params = [v for k, v in self.featurizer.named_parameters() if 'alpha' in k]
            beta_params = [v for k, v in self.featurizer.named_parameters() if 'beta' in k]
            map_params = []
            if 'alpha' in self.hparams['map_opt']:
                map_params.append({'params': alpha_params, 'lr': self.hparams['lr_alpha']})
            if 'beta' in self.hparams['map_opt']:
                map_params.append({'params': beta_params, 'lr': self.hparams["lr_beta"]})
            self.map_optimizer = torch.optim.Adadelta(map_params)

    @staticmethod
    def neum(v, model, batch):
        def hvp(y, w, v):

            # First backprop
            first_grads = autograd.grad(y, w, retain_graph=True, create_graph=True, allow_unused=True)
            first_grads = torch.nn.utils.parameters_to_vector(first_grads)
            # Elementwise products
            elemwise_products = first_grads @ v
            # Second backprop
            return_grads = autograd.grad(elemwise_products, w, create_graph=True)
            return_grads = torch.nn.utils.parameters_to_vector(return_grads)
            return return_grads

        v = v.detach()
        h_estimate = v
        cnt = 0.
        model.eval()
        iter = 10
        for i in range(iter):
            model.weight.grad *= 0
            y = model(batch[0].detach())
            loss = F.cross_entropy(y, batch[1].detach())
            hv = hvp(loss, model.weight, v)
            v -= hv
            v = v.detach()
            h_estimate = v + h_estimate
            h_estimate = h_estimate.detach()
            # not converge
            if torch.max(abs(h_estimate)) > 10:
                break
            cnt += 1

        model.train()
        return h_estimate.detach()

    def update(self, minibatches, unlabeled=None, return_loss=False):
        loss_swap = 0.0
        trm = 0.0

        if self.update_count >= self.hparams['iters']:
            # TRM
            if self.hparams['class_balanced']:
                # for stability when facing unbalanced labels across environments
                for classifier in self.clist:
                    classifier.weight.data = copy.deepcopy(self.classifier.weight.data)
            self.alpha /= self.alpha.sum(1, keepdim=True)

            self.featurizer.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer(all_x)
            # updating original network
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

            for i in range(30):
                all_logits_idx = 0
                loss_erm = 0.
                for j, (x, y) in enumerate(minibatches):
                    # j-th domain
                    feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                    all_logits_idx += x.shape[0]
                    loss_erm += F.cross_entropy(self.clist[j](feature.detach()), y)
                for opt in self.olist:
                    opt.zero_grad()
                loss_erm.backward()
                for opt in self.olist:
                    opt.step()

            # collect (feature, y)
            feature_split = list()
            y_split = list()
            all_logits_idx = 0
            for i, (x, y) in enumerate(minibatches):
                feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                all_logits_idx += x.shape[0]
                feature_split.append(feature)
                y_split.append(y)

            # estimate transfer risk
            for Q, (x, y) in enumerate(minibatches):
                sample_list = list(range(len(minibatches)))
                sample_list.remove(Q)

                loss_Q = F.cross_entropy(self.clist[Q](feature_split[Q]), y_split[Q])
                grad_Q = autograd.grad(loss_Q, self.clist[Q].weight, create_graph=True)
                vec_grad_Q = nn.utils.parameters_to_vector(grad_Q)

                loss_P = [F.cross_entropy(self.clist[Q](feature_split[i]), y_split[i])*(self.alpha[Q, i].data.detach())
                          if i in sample_list else 0. for i in range(len(minibatches))]
                loss_P_sum = sum(loss_P)
                grad_P = autograd.grad(loss_P_sum, self.clist[Q].weight, create_graph=True)
                vec_grad_P = nn.utils.parameters_to_vector(grad_P).detach()
                vec_grad_P = self.neum(vec_grad_P, self.clist[Q], (feature_split[Q], y_split[Q]))

                loss_swap += loss_P_sum - self.hparams['cos_lambda'] * (vec_grad_P.detach() @ vec_grad_Q)

                for i in sample_list:
                    self.alpha[Q, i] *= (self.hparams["groupdro_eta"] * loss_P[i].data).exp()

            loss_swap /= len(minibatches)
            trm /= len(minibatches)
        else:
            # ERM
            self.featurizer.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer(all_x)
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

        nll = loss.item()
        self.optimizer_c.zero_grad()
        self.optimizer_f.zero_grad()
        if self.update_count >= self.hparams['iters']:
            loss_swap = (loss + loss_swap)
        else:
            loss_swap = loss

        loss_swap.backward()
        if return_loss:
            return
        else:
            self.optimizer_f.step()
            self.optimizer_c.step()

            loss_swap = loss_swap.item() - nll
            self.update_count += 1

            return {'nll': nll, 'trm_loss': loss_swap}
    
    def update_map(self, minibatches, map_minibatches, unlabeled=None):
        train_x = torch.cat([x for x, y in minibatches])
        train_y = torch.cat([y for x, y in minibatches])
        val_x = torch.cat([x for x, y in map_minibatches])
        val_y = torch.cat([y for x, y in map_minibatches])
        #! 1
        for step in range(self.hparams['adapter_steps']):
            switch_to_ood(self.featurizer)
            switch_to_ood(self.classifier)
            for c in self.clist:
                switch_to_ood(c)
            train_logits = self.classifier(self.featurizer(train_x))
            loss_map = F.cross_entropy(train_logits, train_y)
            self.map_optimizer.zero_grad()
            loss_map.backward()
            self.map_optimizer.step()
        #! 2
        switch_to_finetune(self.featurizer)
        switch_to_finetune(self.classifier)
        for c in self.clist:
            switch_to_finetune(c)
        step_vals = self.update(minibatches)
        #! 3
        switch_to_bilevel(self.featurizer)
        switch_to_bilevel(self.classifier)
        for c in self.clist:
            switch_to_bilevel(c)
        self.update(minibatches, return_loss=True)
       
        def grad2vec(parameters):
            grad_vec = []
            for param in parameters:
                grad_vec.append(param.grad.view(-1).detach())
            return torch.cat(grad_vec)
    
        param_grad_vec = grad2vec(self.featurizer.parameters())
        #! 4
        switch_to_ood(self.featurizer)
        switch_to_ood(self.classifier)
        for c in self.clist:
            switch_to_ood(c)
        self.map_optimizer.zero_grad()
        val_logits = self.classifier(self.featurizer(val_x))
        loss_map = F.cross_entropy(val_logits, val_y)
        loss_map.backward()
        map_grad_vec = grad2vec(self.featurizer.parameters())
        implicit_gradient = - self.hparams['lr2'] * map_grad_vec * param_grad_vec
        
        def append_grad_to_vec(vec, parameters):
            if not isinstance(vec, torch.Tensor):
                raise TypeError('expected torch.Tensor, but got: {}'
                                .format(torch.typename(vec)))
            pointer = 0
            for param in parameters:
                num_param = param.numel()
                param.grad.copy_(param.grad + vec[pointer:pointer + num_param].view_as(param).data)
                pointer += num_param

        append_grad_to_vec(implicit_gradient, self.featurizer.parameters())
        self.map_optimizer.step()
        # self.optimizer.step()

        return {'nll': step_vals['nll'], 'trm_loss': step_vals['trm_loss']}
    
    def predict(self, x):
        return self.classifier(self.featurizer(x))

    def train(self):
        self.featurizer.train()

    def eval(self):
        self.featurizer.eval()

#! IB_ERM
class IB_ERM(ERM):
    """Information Bottleneck based ERM on feature with conditionning"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IB_ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None, return_loss=False):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                          >= self.hparams['ib_penalty_anneal_iters'] else
                          0.0)

        nll = 0.
        ib_penalty = 0.

        all_x = torch.cat([x for x,y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['ib_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        if return_loss:
            return
        else:
            self.optimizer.step()

            self.update_count += 1
            return {'loss': loss.item(),
                    'nll': nll.item(),
                    'IB_penalty': ib_penalty.item()}
    
    def update_map(self, minibatches, map_minibatches, unlabeled=None):
        train_x = torch.cat([x for x, y in minibatches])
        train_y = torch.cat([y for x, y in minibatches])
        val_x = torch.cat([x for x, y in map_minibatches])
        val_y = torch.cat([y for x, y in map_minibatches])
        #! 1
        for step in range(self.hparams['adapter_steps']):
            switch_to_ood(self.network)
            train_features = self.featurizer(train_x)
            train_logits = self.classifier(train_features)
            loss_map = F.cross_entropy(train_logits, train_y)
            self.map_optimizer.zero_grad()
            loss_map.backward()
            self.map_optimizer.step()
        #! 2     
        switch_to_finetune(self.network) 
        step_vals = self.update(minibatches)
        #! 3
        switch_to_bilevel(self.network) 
        self.update(minibatches, return_loss=True)
    
        def grad2vec(parameters):
            grad_vec = []
            for param in parameters:
                grad_vec.append(param.grad.view(-1).detach())
            return torch.cat(grad_vec)
    
        param_grad_vec = grad2vec(self.network.parameters())
        #! 4
        switch_to_ood(self.network)
        self.map_optimizer.zero_grad()
        val_logits = self.classifier(self.featurizer(val_x))
        loss_map = F.cross_entropy(val_logits, val_y)
        loss_map.backward()
        map_grad_vec = grad2vec(self.network.parameters())
        implicit_gradient = - self.hparams['lr2'] * map_grad_vec * param_grad_vec
        
        def append_grad_to_vec(vec, parameters):
            if not isinstance(vec, torch.Tensor):
                raise TypeError('expected torch.Tensor, but got: {}'
                                .format(torch.typename(vec)))
            pointer = 0
            for param in parameters:
                num_param = param.numel()
                param.grad.copy_(param.grad + vec[pointer:pointer + num_param].view_as(param).data)
                pointer += num_param

        append_grad_to_vec(implicit_gradient, self.network.parameters())
        self.map_optimizer.step()
        # self.optimizer.step()
       
        return {'loss': step_vals['loss'],
                'nll': step_vals['nll'],
                'IB_penalty': step_vals['IB_penalty']}
        
#! IB_IRM
class IB_IRM(ERM):
    """Information Bottleneck based IRM on feature with conditionning"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IB_IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None, return_loss=False):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        irm_penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                          >= self.hparams['ib_penalty_anneal_iters'] else
                          0.0)

        nll = 0.
        irm_penalty = 0.
        ib_penalty = 0.

        all_x = torch.cat([x for x,y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            irm_penalty += self._irm_penalty(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        irm_penalty /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += irm_penalty_weight * irm_penalty
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['irm_penalty_anneal_iters'] or self.update_count == self.hparams['ib_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        if return_loss:
            return
        else:
            self.optimizer.step()

            self.update_count += 1
            return {'loss': loss.item(),
                    'nll': nll.item(),
                    'IRM_penalty': irm_penalty.item(),
                    'IB_penalty': ib_penalty.item()}
            
    def update_map(self, minibatches, map_minibatches, unlabeled=None):
        self.network.train()
        train_x = torch.cat([x for x, y in minibatches])
        train_y = torch.cat([y for x, y in minibatches])
        val_x = torch.cat([x for x, y in map_minibatches])
        val_y = torch.cat([y for x, y in map_minibatches])
        #! 1
        for step in range(self.hparams['adapter_steps']):
            switch_to_ood(self.network)
            train_logits = self.classifier(self.featurizer(train_x))
            loss_map = F.cross_entropy(train_logits, train_y)
            self.map_optimizer.zero_grad()
            loss_map.backward()
            self.map_optimizer.step()
        #! 2 
        switch_to_finetune(self.network) 
        step_vals = self.update(minibatches)
        #! 3
        switch_to_bilevel(self.network) 
        self.update(minibatches, return_loss=True)
        
        def grad2vec(parameters):
            grad_vec = []
            for param in parameters:
                grad_vec.append(param.grad.view(-1).detach())
            return torch.cat(grad_vec)
    
        param_grad_vec = grad2vec(self.network.parameters())
        #! 4
        switch_to_ood(self.network)
        self.map_optimizer.zero_grad()
        val_logits = self.network(val_x)
        loss_map = F.cross_entropy(val_logits, val_y)
        loss_map.backward()
        map_grad_vec = grad2vec(self.network.parameters())
        implicit_gradient = - self.hparams['lr2'] * map_grad_vec * param_grad_vec
        
        def append_grad_to_vec(vec, parameters):
            if not isinstance(vec, torch.Tensor):
                raise TypeError('expected torch.Tensor, but got: {}'
                                .format(torch.typename(vec)))
            pointer = 0
            for param in parameters:
                num_param = param.numel()
                param.grad.copy_(param.grad + vec[pointer:pointer + num_param].view_as(param).data)
                pointer += num_param

        append_grad_to_vec(implicit_gradient, self.network.parameters())
        self.map_optimizer.step()
        # self.optimizer.step()
         
        return {'loss': step_vals['loss'],
                'nll': step_vals['nll'],
                'IRM_penalty': step_vals['IRM_penalty'],
                'IB_penalty': step_vals['IB_penalty']},
        
#! AbstractCAD
class AbstractCAD(Algorithm):
    """Contrastive adversarial domain bottleneck (abstract class)
    from Optimal Representations for Covariate Shift <https://arxiv.org/abs/2201.00057>
    """
    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, is_conditional, feature_dim=128):
        super(AbstractCAD, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        params = list(self.featurizer.parameters()) + list(self.classifier.parameters())

        # parameters for domain bottleneck loss
        self.is_conditional = is_conditional  # whether to use bottleneck conditioned on the label
        self.base_temperature = 0.07
        self.temperature = hparams['temperature']
        self.is_project = hparams['is_project']  # whether apply projection head
        self.is_normalized = hparams['is_normalized'] # whether apply normalization to representation when computing loss

        # whether flip maximize log(p) (False) to minimize -log(1-p) (True) for the bottleneck loss
        # the two versions have the same optima, but we find the latter is more stable
        self.is_flipped = hparams["is_flipped"]

        if self.is_project:
            self.project = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, 128),
            )
            params += list(self.project.parameters())

        # Optimizers
        self.optimizer = torch.optim.Adam(
            params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        # optimizer for bi-level
        if "used_map" in self.hparams and self.hparams["used_map"] == True:
            reset_model(self.featurizer, self.hparams['map_init'], self.hparams["map_ad_type"])
            alpha_params = [v for k, v in self.featurizer.named_parameters() if 'alpha' in k]
            beta_params = [v for k, v in self.featurizer.named_parameters() if 'beta' in k]
            map_params = []
            if 'alpha' in self.hparams['map_opt']:
                map_params.append({'params': alpha_params, 'lr': self.hparams['lr_alpha']})
            if 'beta' in self.hparams['map_opt']:
                map_params.append({'params': beta_params, 'lr': self.hparams["lr_beta"]})
            self.map_optimizer = torch.optim.Adadelta(map_params)
        
    def bn_loss(self, z, y, dom_labels):
        """Contrastive based domain bottleneck loss
         The implementation is based on the supervised contrastive loss (SupCon) introduced by
         P. Khosla, et al., in “Supervised Contrastive Learning“.
        Modified from  https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/losses.py#L11
        """
        device = z.device
        batch_size = z.shape[0]

        y = y.contiguous().view(-1, 1)
        dom_labels = dom_labels.contiguous().view(-1, 1)
        mask_y = torch.eq(y, y.T).to(device)
        mask_d = (torch.eq(dom_labels, dom_labels.T)).to(device)
        mask_drop = ~torch.eye(batch_size).bool().to(device)  # drop the "current"/"self" example
        mask_y &= mask_drop
        mask_y_n_d = mask_y & (~mask_d)  # contain the same label but from different domains
        mask_y_d = mask_y & mask_d  # contain the same label and the same domain
        mask_y, mask_drop, mask_y_n_d, mask_y_d = mask_y.float(), mask_drop.float(), mask_y_n_d.float(), mask_y_d.float()

        # compute logits
        if self.is_project:
            z = self.project(z)
        if self.is_normalized:
            z = F.normalize(z, dim=1)
        outer = z @ z.T
        logits = outer / self.temperature
        logits = logits * mask_drop
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        if not self.is_conditional:
            # unconditional CAD loss
            denominator = torch.logsumexp(logits + mask_drop.log(), dim=1, keepdim=True)
            log_prob = logits - denominator

            mask_valid = (mask_y.sum(1) > 0)
            log_prob = log_prob[mask_valid]
            mask_d = mask_d[mask_valid]

            if self.is_flipped:  # maximize log prob of samples from different domains
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (~mask_d).float().log(), dim=1)
            else:  # minimize log prob of samples from same domain
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (mask_d).float().log(), dim=1)
        else:
            # conditional CAD loss
            if self.is_flipped:
                mask_valid = (mask_y_n_d.sum(1) > 0)
            else:
                mask_valid = (mask_y_d.sum(1) > 0)

            mask_y = mask_y[mask_valid]
            mask_y_d = mask_y_d[mask_valid]
            mask_y_n_d = mask_y_n_d[mask_valid]
            logits = logits[mask_valid]

            # compute log_prob_y with the same label
            denominator = torch.logsumexp(logits + mask_y.log(), dim=1, keepdim=True)
            log_prob_y = logits - denominator

            if self.is_flipped:  # maximize log prob of samples from different domains and with same label
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_n_d.log(), dim=1)
            else:  # minimize log prob of samples from same domains and with same label
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_d.log(), dim=1)

        def finite_mean(x):
            # only 1D for now
            num_finite = (torch.isfinite(x).float()).sum()
            mean = torch.where(torch.isfinite(x), x, torch.tensor(0.0).to(x)).sum()
            if num_finite != 0:
                mean = mean / num_finite
            else:
                return torch.tensor(0.0).to(x)
            return mean

        return finite_mean(bn_loss)

    def update(self, minibatches, unlabeled=None, return_loss=False):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        all_d = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        bn_loss = self.bn_loss(all_z, all_y, all_d)
        clf_out = self.classifier(all_z)
        clf_loss = F.cross_entropy(clf_out, all_y)
        total_loss = clf_loss + self.hparams['lmbda'] * bn_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        if return_loss:
            return 
        else:
            self.optimizer.step()

            return {"clf_loss": clf_loss.item(), 
                    "bn_loss": bn_loss.item(), 
                    "total_loss": total_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

#! CAD
class CAD(AbstractCAD):
    """Contrastive Adversarial Domain (CAD) bottleneck

       Properties:
       - Minimize I(D;Z)
       - Require access to domain labels but not task labels
       """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CAD, self).__init__(input_shape, num_classes, num_domains, hparams, is_conditional=False)

#! CondCAD
class CondCAD(AbstractCAD):
    """Conditional Contrastive Adversarial Domain (CAD) bottleneck

    Properties:
    - Minimize I(D;Z|Y)
    - Require access to both domain labels and task labels
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CondCAD, self).__init__(input_shape, num_classes, num_domains, hparams, is_conditional=True)
        
#! AbstractCausIRL
class AbstractCausIRL(ERM):
    '''Abstract class for Causality based invariant representation learning algorithm from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractCausIRL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None, return_loss=False):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        first = None
        second = None

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i] + 1e-16, targets[i])
            slice = np.random.randint(0, len(features[i]))
            if first is None:
                first = features[i][:slice]
                second = features[i][slice:]
            else:
                first = torch.cat((first, features[i][:slice]), 0)
                second = torch.cat((second, features[i][slice:]), 0)
        if len(first) > 1 and len(second) > 1:
            penalty = torch.nan_to_num(self.mmd(first, second))
        else:
            penalty = torch.tensor(0)
        objective /= nmb

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        if return_loss:
            return 
        else:
            self.optimizer.step()

            if torch.is_tensor(penalty):
                penalty = penalty.item()

            return {'loss': objective.item(), 
                    'penalty': penalty}

#! CausIRL_CORAL
class CausIRL_CORAL(AbstractCausIRL):
    '''Causality based invariant representation learning algorithm using the CORAL distance from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CausIRL_CORAL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, gaussian=False)