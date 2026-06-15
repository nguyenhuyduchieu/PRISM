# -*- coding: utf-8 -*-
"""
Mamba block implementation with selective state space modeling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from .normalization import RMSNorm


class MambaBlock(nn.Module):
    """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper."""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Input projections
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.in_proj_r = nn.Linear(args.d_model, args.d_inner, bias=args.bias)
        
        # 1D convolution
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )
        
        # State space parameters projection
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        # Normalization and output projection
        self.norm_f = RMSNorm(args.d_model)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        
        # Delta projection
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        
        # State space parameters
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
    
    def forward(self, x):
        """Mamba block forward pass."""
        (b, l, d) = x.shape
        
        # Input projection
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        
        # 1D convolution
        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        # Activation
        x = F.silu(x)
        gate = x * (1 - F.sigmoid(res))
        
        # State space modeling
        y = self.ssm(x)
        y = y * F.silu(res)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
    
    def ssm(self, x):
        """Runs the SSM (State Space Model)."""
        (d_in, n) = self.A_log.shape
        
        # Compute state space parameters
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()
        
        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)
        
        return y
    
    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm."""
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # Perform selective scan
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
        
        return y
