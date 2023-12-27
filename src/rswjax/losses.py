import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
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

class LeastSquaresLoss():

    def __init__(self, fdes, diag_weight=None):
        if isinstance(fdes, Number):
            fdes = jnp.array([fdes])
        self.fdes = fdes
        self.m = fdes.size
        if diag_weight is None:
            diag_weight = 1.
        self.diag_weight = diag_weight

    def prox(self, f, lam):
        return (self.diag_weight**2 * self.fdes + f / lam) / (self.diag_weight**2 + 1 / lam)

    def evaluate(self, f):
        return jnp.sum(jnp.square(self.diag_weight * (f - self.fdes)))


def _entropy_prox(f, lam):
    # I'm hopeful this'll become native soon via https://github.com/google/jax/issues/13680;
    # in the meantime this will do
    return lam * jnp.real(tfp.lambertw(jnp.exp(f / lam - 1) / lam, tol=1e-10))