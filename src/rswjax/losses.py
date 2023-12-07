import jax.numpy as jnp
from numbers import Number

class EqualityLoss():

    def __init__(self, fdes):
        if isinstance(fdes, Number):
            fdes = jnp.array([fdes])
        self.fdes = fdes
        self.m = fdes.size

    def prox(self, f, lam):
        return self.fdes