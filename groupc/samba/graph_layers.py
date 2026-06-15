# -*- coding: utf-8 -*-
"""
Graph neural network layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class gconv(nn.Module):
    """Graph convolution layer using Chebyshev polynomials"""
    
    def __init__(self, inp, hid, embed, cheb_k, n):
        super(gconv, self).__init__()
        
        self.node_num = n
        self.inp = inp
        self.cheb_k = cheb_k
        
        # Learnable adjacency matrix
        self.adj = nn.Parameter(torch.randn(n, embed), requires_grad=True)
        
        # Chebyshev polynomial weights
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed, cheb_k, inp, hid))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed, hid))
    
    def forward(self, x):
        """
        Args:
            x: shape [B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        Returns:
            output: shape [B, N, C]
        """
        # Generate adjacency matrix
        ADJ = F.softmax(F.relu(torch.mm(self.adj, self.adj.transpose(0, 1))), dim=1)
        
        # Build Chebyshev polynomial support set
        support_set = [torch.eye(self.node_num).cuda(), ADJ]
        
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * ADJ, support_set[-1]) - support_set[-2])
        
        supports = torch.stack(support_set, dim=0)
        
        # Apply Chebyshev graph convolution
        weights = torch.einsum('nd,dkio->nkio', self.adj, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(self.adj, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        out_6 = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # B, N, D_OUT
        
        return out_6


class AVWGCN(nn.Module):
    """Adaptive Vertex-wise Graph Convolutional Network"""
    
    def __init__(self, dim_in, hid, cheb_k, n):
        super(AVWGCN, self).__init__()
        
        self.node_num = n
        self.inp = dim_in
        self.cheb_k = cheb_k
        
        # Node embeddings for adaptive adjacency matrix
        self.node_embeddings = nn.Parameter(torch.randn(n, dim_in, dim_in), requires_grad=True)
        
        # Chebyshev polynomial weights
        self.weights_pool = nn.Parameter(torch.FloatTensor(cheb_k, n, dim_in, hid))
        self.bias_pool = nn.Parameter(torch.FloatTensor(n, hid))
    
    def forward(self, x):
        """
        Args:
            x: shape [B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        Returns:
            output: shape [B, N, C]
        """
        # Generate adaptive supports
        supports = F.softmax(F.relu(self.node_embeddings), dim=2)
        
        I = torch.eye(self.inp).cuda()
        I2 = I[None, :, :].repeat(x.size(1), 1, 1)
        
        support_set = [I2, supports]
        supports = torch.stack(support_set, dim=0)
        
        # Apply graph convolution
        x_g = torch.einsum("bnc,kncm->bknm", x, supports)  # B, cheb_k, N, dim_in
        x_gconv = torch.einsum('bknm,knmo->bno', x_g, self.weights_pool) + self.bias_pool  # b, N, dim_out
        
        return x_gconv
