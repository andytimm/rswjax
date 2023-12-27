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
    
class InequalityLoss():

    def __init__(self, fdes, lower, upper):
        if isinstance(fdes, Number):
            fdes = jnp.array([fdes])
        self.fdes = fdes
        self.m = fdes.size
        self.lower = lower
        self.upper = upper
        assert (self.lower <= self.upper).all()

    def prox(self, f, lam):
        return jnp.clip(f, self.fdes + self.lower, self.fdes + self.upper)