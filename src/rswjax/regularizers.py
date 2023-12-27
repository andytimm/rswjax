import jax.numpy as jnp
import tensorflow_probability.substrates.jax.math as tfp

class ZeroRegularizer():

    def __init__(self):
        pass

    def prox(self, w, lam):
        return w

class EntropyRegularizer():

    def __init__(self, limit=None):
        if limit is not None and limit <= 1:
            raise ValueError(f"limit is {limit:.3f}. It must be > 1.")
        self.limit = limit

    def prox(self, w, lam):
        what = lam * jnp.real(tfp.lambertw(jnp.exp(w / lam - 1) / lam))
        if self.limit is not None:
            what = jnp.clip(what, 1 / (self.limit * w.size),
                           self.limit / w.size)
        return what
