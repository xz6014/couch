"""
Class of ExpertWeights
"""
import numpy as np
import torch
import torch.nn as nn


class ExpertWeights(nn.Module):
    def __init__(self, rng, shape):
        super(ExpertWeights, self).__init__()

        """rng"""
        self.initialRNG = rng

        """shape"""
        self.weight_shape = shape  # 4/8 * out * in
        self.bias_shape = (shape[0], shape[1], 1)  # 4/8 * out * 1

        """alpha and beta"""
        self.register_parameter(name='alpha', param=self.initial_alpha())
        self.register_parameter(name='beta', param=self.initial_beta())

        # self.alpha = torch.Variable(self.initial_alpha(), name=name + 'alpha')
        # self.beta = nn.Variable(self.initial_beta(), name=name + 'beta')

    """initialize parameters for experts i.e. alpha and beta"""

    def initial_alpha_np(self):
        shape = self.weight_shape
        rng = self.initialRNG
        alpha_bound = np.sqrt(6. / np.prod(shape[-2:]))
        alpha = np.asarray(
            rng.uniform(low=-alpha_bound, high=alpha_bound, size=shape),
            dtype=np.float32)
        return alpha

    def initial_alpha(self):
        alpha = self.initial_alpha_np()
        return torch.nn.Parameter(torch.from_numpy(alpha).float().requires_grad_())

    def initial_beta(self):
        return torch.nn.Parameter(torch.zeros(self.bias_shape).float().requires_grad_())

    def weight(self, x, controlweights):
        weight = torch.einsum('ijk,il->ljk', self.alpha, controlweights)
        bias = torch.einsum('ijk,il->ljk', self.beta, controlweights)
        x = torch.bmm(weight, x) + bias
        # out = torch.bmm(torch.einsum('ijk,il->ljk', self.alpha, controlweights), x) + \
        #     torch.einsum('ijk,il->ljk', self.beta, controlweights)
        return x

    def get_NNweight(self, controlweights, batch_size):
        r = torch.einsum('ijk,il->ljk', self.alpha, controlweights)
        return r  # ?*out*1


    def get_NNbias(self, controlweights, batch_size):
        r = torch.einsum('ijk,il->ljk', self.beta, controlweights)
        return r  # ?*out*1

