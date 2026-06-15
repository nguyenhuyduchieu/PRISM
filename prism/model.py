
import torch
import torch.nn as nn
import torch.nn.functional as F


class PRISMConfig:

    def __init__(self):
        # IO
        self.seq_len = 96
        self.pred_len = 96
        self.num_nodes = 5  # number of assets
        self.batch_size = 32
        
        # Training
        self.learning_rate = 1.5e-3
        self.weight_decay = 1e-4
        self.epochs = 40
        
        # Regime encoder
        self.num_regimes = 4
        self.regime_dim = 64
        self.regime_temp = 1.0
        
        # Graph
        self.graph_hidden = 64
        
        # Frequency bank
        self.num_bands = 5
        self.band_kernel = 15
        
        # HyperLinear
        self.linear_rank = 8
        
        # Probabilistic
        self.quantiles = [0.1, 0.5, 0.9]


class RevIN(nn.Module):
    """Reversible Instance Normalization for distribution shift handling"""
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True) + self.eps
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.gamma + self.beta
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.beta) / self.gamma
            x = x * self.std + self.mean
        return x


class RGRLCore(nn.Module):
    """Reversible Graph-Regularized Linear Core"""
    def __init__(self, seq_len, pred_len, channels, use_revin=True):
        super().__init__()
        self.use_revin = use_revin
        self.revin = RevIN(channels)
        self.linear = nn.Linear(seq_len, pred_len)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # x: [B, L, N]
        if self.use_revin:
            x = self.revin(x, mode='norm')
        x = x.permute(0, 2, 1)  # [B, N, L]
        x = self.linear(x)  # [B, N, S]
        x = x.permute(0, 2, 1)  # [B, S, N]
        if self.use_revin:
            x = self.revin(x, mode='denorm')
        return x


class RegimeEncoder(nn.Module):
    """
    Regime Encoder - Identifies latent market states
    Outputs interpretable regime probabilities via Gumbel-Softmax
    """
    def __init__(self, in_channels, num_regimes=4, latent_dim=64, temperature=1.0):
        super().__init__()
        self.num_regimes = num_regimes
        self.temperature = temperature
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.proj = nn.Linear(64, latent_dim)
        self.logits_head = nn.Linear(latent_dim, num_regimes)
        self.decoder = nn.Linear(latent_dim, 64)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x: [B, L, C] -> [B, C, L]
        h = self.encoder(x.permute(0, 2, 1)).squeeze(-1)
        h = self.dropout(h)
        z = self.proj(h)
        logits = self.logits_head(z)
        gate = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)
        recon = self.decoder(z)
        return z, logits, gate, recon


class DynamicGraph(nn.Module):
    """
    Dynamic Graph - Learns time-evolving cross-asset adjacency matrix
    Provides interpretable graph heatmaps
    """
    def __init__(self, num_nodes, hidden_dim=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        self.raw_A_base = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.01)
        
        self.adj_generator = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_nodes * num_nodes)
        )
        
        self.ctx_mlp = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.dynamic_scale = nn.Parameter(torch.tensor(0.1))
    
    def get_adjacency(self, features):
        B = features.shape[0]
        
        A_base = F.softplus(self.raw_A_base)
        A_base = (A_base + A_base.t()) / 2
        A_base = A_base - torch.diag_embed(torch.diag(A_base))
        
        delta = self.adj_generator(features).view(B, self.num_nodes, self.num_nodes)
        delta = torch.tanh(delta) * self.dynamic_scale
        
        A = A_base.unsqueeze(0) + delta
        mask = 1.0 - torch.eye(self.num_nodes, device=A.device).unsqueeze(0)
        A = A * mask
        
        return A
    
    def forward(self, features):
        ctx = self.ctx_mlp(features)
        A = self.get_adjacency(features)
        return ctx, A


class FrequencyBank(nn.Module):
    """
    Frequency Bank - Multi-scale temporal decomposition
    Provides interpretable frequency gating weights
    """
    def __init__(self, channels, num_bands=5, kernel=15):
        super().__init__()
        self.num_bands = num_bands
        
        self.filters = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=kernel, 
                     padding=kernel//2, groups=channels, bias=False)
            for _ in range(num_bands)
        ])
        
        self.gate = nn.Parameter(torch.ones(num_bands) / num_bands)
        
        for i, filt in enumerate(self.filters):
            nn.init.normal_(filt.weight, mean=0, std=0.1 / (i + 1))
    
    def forward(self, x):
        B, L, N = x.shape
        x_t = x.permute(0, 2, 1)
        
        bands = []
        for filt in self.filters:
            band = filt(x_t)
            bands.append(band.permute(0, 2, 1))
        
        bands = torch.stack(bands, dim=1)
        weights = F.softmax(self.gate, dim=0)
        x_fused = (weights.view(1, -1, 1, 1) * bands).sum(dim=1)
        
        return x_fused, bands, weights


class HyperLinear(nn.Module):
    """
    HyperLinear - Context-adaptive weight generation via low-rank decomposition
    Core innovation: generates sample-specific weights conditioned on context
    """
    def __init__(self, seq_len, pred_len, rank=8, ctx_dim=128):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.rank = rank
        
        self.W_base = nn.Parameter(torch.zeros(pred_len, seq_len))
        nn.init.xavier_uniform_(self.W_base, gain=0.1)
        
        self.hyper = nn.Sequential(
            nn.Linear(ctx_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, rank * (seq_len + pred_len))
        )
        
        self.adapt_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x, ctx):
        B, L, _ = x.shape
        
        h = self.hyper(ctx)
        A_delta = h[:, :self.rank * self.pred_len].view(B, self.pred_len, self.rank)
        B_delta = h[:, self.rank * self.pred_len:].view(B, self.rank, self.seq_len)
        
        delta_W = self.adapt_scale * (A_delta @ B_delta)
        W_eff = self.W_base.unsqueeze(0) + delta_W
        
        y = (W_eff @ x.squeeze(-1).unsqueeze(-1)).squeeze(-1)
        
        return y.unsqueeze(-1), W_eff


class QuantileHead(nn.Module):
    """Quantile Head for probabilistic forecasting"""
    def __init__(self, pred_len, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.linear = nn.Linear(pred_len, pred_len * self.num_quantiles)
    
    def forward(self, x):
        B, S, _ = x.shape
        out = self.linear(x.squeeze(-1))
        out = out.view(B, S, self.num_quantiles)
        out = torch.sort(out, dim=-1)[0]
        return out


class PRISMModel(nn.Module):
    """
    PRISM - Prediction-market Regime-aware Integrated Spatiotemporal Model

    A regime-aware hypernetwork that dynamically generates context-conditioned
    low-rank weight adaptations for multi-asset cryptocurrency forecasting.
    
    Key features:
    - Regime-aware: Detects market states (bull, bear, volatile, sideways)
    - Cross-asset: Learns dynamic inter-asset dependencies
    - Multi-scale: Decomposes temporal patterns across frequency bands
    - Interpretable: Glass-box explainability via auxiliary outputs
    - Probabilistic: Quantile predictions for uncertainty estimation
    """
    
    def __init__(self, config: PRISMConfig = None):
        super().__init__()
        if config is None:
            config = PRISMConfig()
        
        self.cfg = config
        N = config.num_nodes

        # Ablation switches (default False -> full PRISM, numbers unchanged)
        self.abl_no_graph = getattr(config, 'abl_no_graph', False)
        self.abl_no_hyper = getattr(config, 'abl_no_hyper', False)
        self.abl_no_revin = getattr(config, 'abl_no_revin', False)
        self.abl_no_freq = getattr(config, 'abl_no_freq', False)
        self.abl_no_regime = getattr(config, 'abl_no_regime', False)
        self.abl_no_crossmodal = getattr(config, 'abl_no_crossmodal', False)
        # channels that carry the prediction-market (poly_up) signal; used to
        # mask the graph input for the cross-modal ablation. Defaults to the
        # asset-major layout (every 4th channel).
        self.poly_idx = list(getattr(config, 'poly_idx', range(0, N, 4)))

        # Core linear with RevIN
        self.core = RGRLCore(config.seq_len, config.pred_len, channels=N,
                             use_revin=not self.abl_no_revin)
        
        # Regime encoder
        self.regime = RegimeEncoder(
            in_channels=N, 
            num_regimes=config.num_regimes,
            latent_dim=config.regime_dim,
            temperature=config.regime_temp
        )
        
        # Dynamic graph
        self.graph = DynamicGraph(num_nodes=N, hidden_dim=config.graph_hidden)
        
        # Frequency bank
        self.freq = FrequencyBank(
            channels=N, 
            num_bands=config.num_bands,
            kernel=config.band_kernel
        )
        
        # HyperLinear
        ctx_dim = config.regime_dim + config.graph_hidden + config.num_bands
        self.hyper = HyperLinear(
            config.seq_len, config.pred_len, 
            rank=config.linear_rank, ctx_dim=ctx_dim
        )
        
        # Quantile head
        self.qhead = QuantileHead(config.pred_len, quantiles=config.quantiles)
        
        # Residual scaling
        self.residual_scale = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x, return_aux=False):
        """
        Forward pass
        
        Args:
            x: [B, L, N] input tensor (batch, seq_len, num_assets)
            return_aux: whether to return auxiliary outputs for explainability
        
        Returns:
            y_point: [B, S, N] point predictions
            q: [B, S, Q, N] quantile predictions (if return_aux)
            aux: dict of interpretable outputs (if return_aux)
        """
        B, L, N = x.shape

        # === Context Stream ===
        z, logits, gate, recon = self.regime(x)
        if self.abl_no_regime:
            z = torch.zeros_like(z)

        features = x[:, -1, :]
        if self.abl_no_crossmodal:
            # restrict the cross-asset context to the prediction-market
            # channels only, severing the spot -> market routing path
            mask = torch.zeros(N, device=x.device, dtype=x.dtype)
            mask[self.poly_idx] = 1.0
            features = features * mask
        g_ctx, A_learned = self.graph(features)
        if self.abl_no_graph:
            g_ctx = torch.zeros_like(g_ctx)

        x_fused, bands, freq_weights = self.freq(x)
        if self.abl_no_freq:
            x_fused = x  # bypass the multi-scale frequency filtering

        # === Prediction Stream ===
        y_core = self.core(x)

        ctx = torch.cat([
            z, g_ctx,
            freq_weights.unsqueeze(0).expand(B, -1)
        ], dim=-1)

        y_hyper_list = []
        W_list = []
        for i in range(N):
            y_i, W_i = self.hyper(x_fused[:, :, i:i+1], ctx)
            y_hyper_list.append(y_i)
            W_list.append(W_i)

        y_hyper = torch.cat(y_hyper_list, dim=2)
        if self.abl_no_hyper:
            y_point = y_core  # core-only: drop the context-adaptive correction
        else:
            y_point = y_core + self.residual_scale * y_hyper
        
        if return_aux:
            q_list = [self.qhead(y_point[:, :, i:i+1]) for i in range(N)]
            q = torch.stack(q_list, dim=3)
            
            aux = {
                'regime_logits': logits,
                'regime_gate': gate,
                'regime_z': z,
                'graph_A': A_learned,
                'graph_ctx': g_ctx,
                'freq_weights': freq_weights,
                'freq_bands': bands,
                'hyper_W': torch.stack(W_list, dim=3),
            }
            
            return y_point, q, aux
        
        return y_point
    
    def get_interpretable_outputs(self, x):
        """Get all interpretable outputs for glass-box explainability"""
        _, q, aux = self.forward(x, return_aux=True)
        
        return {
            'regime_probabilities': F.softmax(aux['regime_logits'], dim=-1),
            'graph_heatmap': aux['graph_A'],
            'frequency_importance': aux['freq_weights'],
            'quantile_predictions': q,
            'context_vector': torch.cat([aux['regime_z'], aux['graph_ctx']], dim=-1)
        }
