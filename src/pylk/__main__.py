import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Any, Optional, Tuple, Union

Array = Any

@jax.custom_jvp
@jax.jit
def relu(x: Array) -> Array:
    r"""Rectifier activation function

    Computes the unit-wise rectified linear unit activation over x:

    .. math::
        \mathrm{relu}(x) = \max(x, 0)

    For differentiation, we take:

    .. math::
        \nabla \mathrm{relu}(0) = 0

    Inspired heavily by `JAX's relu <https://jax.readthedocs.io/en/latest/_modules/jax/_src/nn/functions.html#relu>`_.

    Args:
      x : input array
    """
    return jnp.maximum(0., x)

@relu.defjvp
def relu_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    primal_out = relu(x)
    tangent_out = x_dot * jnp.maximum(0., x) / x
    return primal_out, tangent_out

def main():
    xs = jnp.arange(-3., 3., step=0.1)
    ys = relu(xs)
    yps = jax.vmap(jax.grad(relu))(xs)
    plt.plot(xs, ys, label="relu(x)")
    plt.plot(xs, yps, label="relu'(x)")
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()
