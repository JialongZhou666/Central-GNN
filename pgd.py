from torch.optim.sgd import SGD
from torch.optim.optimizer import required
from torch.optim import Optimizer
import torch
import sklearn
import numpy as np
import scipy.sparse as sp
import networkx as nx
import math
from betaspace import betaspace_F
import random

class PGD(Optimizer):
    """Proximal gradient descent.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining parameter groups
    proxs : iterable
        iterable of proximal operators
    alpha : iterable
        iterable of coefficients for proximal gradient descent
    lr : float
        learning rate
    momentum : float
        momentum factor (default: 0)
    weight_decay : float
        weight decay (L2 penalty) (default: 0)
    dampening : float
        dampening for momentum (default: 0)

    """

    def __init__(self, params, proxs, alphas, lr=required, momentum=0, dampening=0, weight_decay=0):
        defaults = dict(lr=lr, momentum=0, dampening=0,
                        weight_decay=0, nesterov=False)


        super(PGD, self).__init__(params, defaults)

        for group in self.param_groups:
            group.setdefault('proxs', proxs)
            group.setdefault('alphas', alphas)

    def __setstate__(self, state):
        super(PGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('proxs', proxs)
            group.setdefault('alphas', alphas)

    def step(self, delta=0, closure=None):
         for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            proxs = group['proxs']
            alphas = group['alphas']

            # apply the proximal operator to each parameter in a group
            for param in group['params']:
                for prox_operator, alpha in zip(proxs, alphas):
                    # param.data.add_(lr, -param.grad.data)
                    # param.data.add_(delta)
                    param.data = prox_operator(param.data, alpha=alpha*lr)


class ProxOperators():
    """Proximal Operators.
    """

    def __init__(self):
        self.centrality_loss = None

    def compute_loss(self, data):
        """Compute the loss function based on centrality measures.
        """
        device = data.device

        # Compute Laplacian matrix
        deg = torch.sum(data, dim=1)
        laplacian = torch.diag(deg) - data

        # Betweenness Centrality
        b = torch.sum(laplacian, dim=1)
        max_b = torch.max(b)
        x1 = 0.3 * max_b

        # In-Degree Centrality
        i_d = torch.sum(data, dim=0)
        max_i_d = torch.max(i_d)
        x2 = 0.2 * max_i_d

        # Out-Degree Centrality
        o_d = torch.sum(data, dim=1)
        max_o_d = torch.max(o_d)
        x3 = 0.2 * max_o_d

        # Closeness Centrality
        deg_inv = 1.0 / deg
        c = torch.matmul(deg_inv, laplacian)
        max_c = torch.max(c)
        x4 = 0.3 * max_c

        x5 = x1 + x2 + x3 + x4

        if x5:
            # Dataset == "polblogs"
            x6 = math.log(x5, 0.16)

            # Dataset == "cora_ml"
            # x6 = math.tanh(2 * x5)
        else:
            x6 = 0

        x = torch.tensor([[x6]], dtype=torch.float, device=device)

        β_eff, x_eff = betaspace_F(data, x)

        # Dataset == "polblogs"
        target_β_eff = 81.26355
        target_x_eff = 0.656624743002746

        # Dataset == "cora_ml"
        # target_β_eff = 18.255857
        # target_x_eff = 0.3764410649008544

        # Calculate the loss function
        self.centrality_loss = (β_eff - target_β_eff)**2 + (x_eff - target_x_eff)**2

        return self.centrality_loss

    def generate_neighbor(self, data):
        """Generate a neighboring solution by perturbing the data.
        """
        device = data.device
        perturbation = torch.randn(data.size()).to(device)
        new_data = data + perturbation
        new_data.clamp_(min=0, max=1)
        return new_data

    def simulated_annealing(self, data, alpha, initial_temperature=1.0, cooling_rate=0.95, num_iterations=100):
        """Simulated annealing optimization for prox_centrality function.
        """
        device = data.device
        current_data = data.clone()
        best_data = data.clone()
        current_loss = self.compute_loss(current_data)
        best_loss = current_loss
        temperature = initial_temperature

        for iteration in range(num_iterations):
            # generate new results
            new_data = self.generate_neighbor(current_data)

            # calculate new loss
            new_loss = self.compute_loss(new_data)

            # Accept or reject the new solution based on the temperature and the difference in the loss function
            if new_loss < current_loss or random.random() < math.exp((current_loss - new_loss) / temperature):
                current_data = new_data
                current_loss = new_loss

            # Update the beset result
            if new_loss < best_loss:
                best_data = new_data
                best_loss = new_loss

            # Lower the temprature
            temperature *= cooling_rate

        return best_data


class SGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

prox_operators = ProxOperators()