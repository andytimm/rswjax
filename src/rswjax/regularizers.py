import jax.numpy as jnp
import tensorflow_probability.substrates.jax.math as tfp

class ZeroRegularizer():

    def __init__(self):
        pass

    def prox(self, w, lam):
        return w