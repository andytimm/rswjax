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

class KLRegularizer():

    def __init__(self, prior, limit=None):
        self.prior = prior
        self.entropy_reg = EntropyRegularizer(limit)

    def prox(self, w, lam):
        return self.entropy_reg.prox(w + lam * jnp.log(self.prior), lam)

class BooleanRegularizer():

    def __init__(self, k):
        self.k = k

    def prox(self, w, lam):
        idx_sort = jnp.argsort(w)
        top_k_indices = idx_sort[-self.k:]
        # adhere to jax array immutability
        new_arr = jnp.where(jnp.isin(jnp.arange(len(w)), top_k_indices), 1. / self.k, 0)
        return new_arr