from __future__ import absolute_import

from typing import Sequence

import jax.numpy as jnp
import flax.linen as nn

class SimpleMLP(nn.Module):
    """Simple Multilayer perceptron class with multiple dense layers with 
    relu activation in between layers
    
    This is an example of compact module classes in flax and are useful where 
    the architectures are simple and donot require defining explicit 
    parameters/layers that are reused in the forward pass (maybe more than 
    once even)

    The @linen.compact decorator is used to define a compact linen module 
    where the layers (other submodules) are defined at the runtime. 
    """

    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
        x = inputs
        for i, feat in enumerate(self.features):

            # Notice, we did not explicitly store the dense layer before 
            # calling the forward pass (just assigned a name to it). 
            # Flax will take care of that at the runtime for us. 
            # Thanks to the @linen.compact decorator
            x = nn.Dense(feat, name=f'layers_{i}')(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x
