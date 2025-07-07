import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import plugin

class DECLinear(nn.Module, ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dec_config = None
        self.k_chunk = None
        self.q_residual = None
        self.scales = None
        self.thresholds = None

    @abstractmethod
    def forward(self, x, **kwargs):
        pass

    def load_dec_data(self, q_residual, scales, thresholds):
        assert (q_residual.dtype == torch.int)
        assert (q_residual.shape == (self.in_features, self.out_features // 8))
        assert (scales.dtype == self.dtype)
        assert (scales.shape == (self.out_features,))
        assert (thresholds.dtype == self.dtype)
        assert (thresholds.shape == (self.in_features,))
        self.q_residual = q_residual
        self.scales = scales
        self.thresholds = thresholds

    def create_dec_config(self, dec_context, k_chunk):
        self.k_chunk = k_chunk
        assert self.q_residual is not None
        assert self.scales is not None
        assert self.thresholds is not None

        self.dec_config = plugin.create_dec_config(
            dec_context,
            k_chunk,
            self.q_residual,
            self.scales, 
            self.thresholds
        )

    def update_dec_config(self, dec_context, k_chunk):
        """Update dec_config without re-initializing the zero-copy residuals"""
        self.k_chunk = k_chunk
        assert self.dec_config is not None
        plugin.update_dec_config(
            self.dec_config,
            dec_context,
            k_chunk,
        )
