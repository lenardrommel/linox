"""Array conversion between PyTorch and JAX.

Kept separate from :mod:`linox.interop.skerch` so the conversion rules have one
home: any future interop target that speaks torch reuses these two functions.

Conversion goes through host memory (``.cpu().numpy()`` in one direction,
:func:`numpy.asarray` in the other). That is deliberate -- it is device-agnostic
and needs no version negotiation between the two frameworks. The zero-copy
upgrade path, should same-device throughput ever matter, is DLPack
(:func:`torch.from_dlpack` and :mod:`jax.dlpack`), which only pays off when both
frameworks already sit on the same accelerator.
"""

import jax.numpy as jnp
import numpy as np

try:
    import torch
except ImportError as e:  # pragma: no cover - exercised only without torch
    msg = (
        "linox.interop requires PyTorch, which is not installed. "
        "Install the interop extra: `pip install linox[interop]` "
        "(or `uv sync --group interop`)."
    )
    raise ImportError(msg) from e


def to_jax(x: "torch.Tensor") -> jnp.ndarray:
    """Convert a torch tensor to a JAX array.

    The tensor is detached first: the boundary is not differentiable in either
    framework, and skerch may hand over tensors that are still attached to an
    autograd graph.
    """
    return jnp.asarray(x.detach().cpu().numpy())


def to_torch(y: jnp.ndarray, like: "torch.Tensor") -> "torch.Tensor":
    """Convert a JAX array to a torch tensor matching ``like``.

    The dtype is pinned back to ``like``'s rather than being inherited from
    ``y``. This is load-bearing, not cosmetic: importing linox enables JAX's x64
    mode process-wide, so a float32 input can come back float64 after any
    internal promotion, and skerch assigns these results straight into buffers
    it preallocated at the dtype the caller declared.
    """
    # `np.asarray` on a JAX array can yield a read-only view, which
    # `torch.as_tensor` would either reject or wrap in a tensor that fails on
    # the in-place writes skerch performs. Copy when that is the case.
    arr = np.asarray(y)
    if not arr.flags.writeable:
        arr = arr.copy()
    return torch.as_tensor(arr).to(dtype=like.dtype, device=like.device)
