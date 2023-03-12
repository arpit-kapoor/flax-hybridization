import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

class MLP(nn.Module):

    n_layers: int
    n_features: list[int]

    def setup(self) -> None:
        self.layers = [nn.Dense(self.n_features[idx]) for idx in range(self.n_layers)]
    
    def __call__(self, x:jnp.ndarray) -> jnp.ndarray:
        out = x
        for layer in self.layers:
            out = nn.relu(layer(out))
        return out
    

class SimpleMLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, name=f'layers_{i}')(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x
