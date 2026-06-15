# -*- coding: utf-8 -*-
"""
Model configuration classes
"""

import math
from dataclasses import dataclass
from typing import Union, List, Dict, Any


@dataclass
class ModelArgs:
    """Configuration for SAMBA model architecture"""
    d_model: int
    n_layer: int
    vocab_size: int
    seq_in: int
    seq_out: int
    d_state: int = 128
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 3
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    # Dataset parameters
    dataset: str = 'STOCK_DATA'
    lag: int = 5
    horizon: int = 1
    num_nodes: int = 82  # 82 daily stock features as per the paper
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Model parameters
    input_dim: int = 1
    output_dim: int = 1
    embed_dim: int = 10
    rnn_units: int = 128
    num_layers: int = 3
    cheb_k: int = 3
    d_in: int = 32
    hid: int = 32
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 1100
    lr_init: float = 0.001
    lr_decay: bool = True
    lr_decay_rate: float = 0.5
    lr_decay_step: List[int] = None
    early_stop: bool = True
    early_stop_patience: int = 200
    grad_norm: bool = False
    max_grad_norm: float = 5
    
    # Loss and metrics
    loss_func: str = 'mae'
    mae_thresh: float = None
    mape_thresh: float = 0
    
    # System parameters
    device: str = 'cuda:0'
    seed: int = 1
    debug: bool = True
    log_step: int = 20
    log_dir: str = './'
    
    def __post_init__(self):
        if self.lr_decay_step is None:
            self.lr_decay_step = [40, 70, 100]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'dataset': self.dataset,
            'mode': 'train',
            'device': self.device,
            'debug': self.debug,
            'model': 'SAMBA',
            'cuda': True,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'lag': self.lag,
            'horizon': self.horizon,
            'num_nodes': self.num_nodes,
            'tod': False,
            'normalizer': 'std',
            'column_wise': False,
            'default_graph': True,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embed_dim': self.embed_dim,
            'rnn_units': self.rnn_units,
            'num_layers': self.num_layers,
            'cheb_k': self.cheb_k,
            'loss_func': self.loss_func,
            'seed': self.seed,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'lr_init': self.lr_init,
            'lr_decay': self.lr_decay,
            'lr_decay_rate': self.lr_decay_rate,
            'lr_decay_step': self.lr_decay_step,
            'early_stop': self.early_stop,
            'early_stop_patience': self.early_stop_patience,
            'grad_norm': self.grad_norm,
            'max_grad_norm': self.max_grad_norm,
            'real_value': False,
            'mae_thresh': self.mae_thresh,
            'mape_thresh': self.mape_thresh,
            'log_dir': self.log_dir,
            'log_step': self.log_step,
            'plot': False,
            'teacher_forcing': False,
            'd_in': self.d_in,
            'hid': self.hid
        }
