from __future__ import absolute_import

import numpy as np
import jax.numpy as jnp

from flax import linen as nn

from .awbm import SimpleAWBM
from .mlp import SimpleMLP

class HybridAWBM(nn.Module):
    
    # AWBM Attributes
    S_init: float
    B_init: float
    
    # MLP Attributes
    n_layers: int
    n_features: list[int]

    def setup(self):
        """Function to setup class attributes
        """
        # Process-based model
        self._process = SimpleAWBM(self.S_init, self.B_init)
        
        # Deep Learning model
        self._mlp = SimpleMLP(self.n_features)
        
    def __call__(self, x):
        """Simulate streamflow using hybrid model

        Args
        ----
        X: 2D input matrix (`batch_size`x 2)
        """
        # TODO: Write the forward pass for the hybrid model
        out = 
        return out

