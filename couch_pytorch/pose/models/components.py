import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experts_weights import ExpertWeights


class ComponentNN(nn.Module):
    def __init__(self, rng, num_experts, dim_layers, activation, keep_prob, name):
        super(ComponentNN, self).__init__()

        """
        :param rng: random seed for numpy
        :param input_x: input tensor of ComponentNN/GatingNN
        :param num_experts: number of experts
        :param dim_layers: dimension of each layer including the dimension of input and output
        :param activation: activation function of each layer
        :param weight_blend: blending weights from previous ComponentNN that used experts in
                             current Components NN.
                             Note that the VanillaNN can also be represented as Components NN with 1 Expert
        :param keep_prob: for drop out
        :param batchSize: for batch size
        :param name: for name of current component
        # :param FiLM: Technique of FiLM, will not use this one in default
        """
        self.name = name

        """rng"""
        self.initialRNG = rng

        """dropout"""
        self.keep_prob = keep_prob

        """"NN structure"""
        self.num_experts = num_experts
        self.dim_layers = dim_layers
        self.num_layers = len(dim_layers) - 1
        self.activation = activation

        """Build NN"""
        self.experts = self.initExperts()
        self.dp_list = []
        for i in range(self.num_layers):
            dp_layer = nn.Dropout(1 - self.keep_prob)
            self.dp_list.append(dp_layer)

    def initExperts(self):
        experts = []
        for i in range(self.num_layers):
            setattr(self, 'experts_layer{}'.format(i),
                    ExpertWeights(self.initialRNG, (self.num_experts, self.dim_layers[i + 1], self.dim_layers[i])))
            experts.append(getattr(self, 'experts_layer{}'.format(i)))
        return experts

    def forward(self, input, weight_blend=None):
        batch_size = input.shape[0]
        H = input.unsqueeze(-1)
        H = F.dropout2d(H, p=1-self.keep_prob, training=False)
        for i in range(self.num_layers):
            if weight_blend is not None:
                w = self.experts[i].get_NNweight(weight_blend, batch_size)
                b = self.experts[i].get_NNbias(weight_blend, batch_size)
            else:
                w = self.experts[i].get_NNweight(torch.ones((1, batch_size)).cuda(), batch_size)
                b = self.experts[i].get_NNbias(torch.ones((1, batch_size)).cuda(), batch_size)
            H = torch.matmul(w, H) + b
            # if weight_blend is not None:
            #     H = self.experts[i].weight(H, weight_blend)
            # else:
            #     H = self.experts[i].weight(H, torch.ones((1, batch_size)).cuda())
            # H = torch.matmul(w, H) + b
            # print(H.shape)

            if i < (self.num_layers - 1):
                H = self.activation[i](H)
                H = F.dropout2d(H, p=1-self.keep_prob, training=False)
            else:
                H = H.squeeze(-1)
                if len(self.activation) > i:
                    H = self.activation[i](H)
        return H
