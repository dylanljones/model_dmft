# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-03-19

from triqs.gf import BlockGf, Gf

from .utility import GfLike


def apply_mixing(old: GfLike, new: GfLike, mixing: float = 1.0) -> GfLike:
    """Apply mixing to a Green's function object.

    Parameters
    ----------
    old : Gf or BlockGf
        The old value of the quantity.
    new : Gf or BlockGf
        The new value of the quantity. Will be overwriten with the result!
    mixing: float
        The mixing value. If `mixing=1` no mixing is applied.

    Returns
    -------
    Gf or BlockGf
        The mixed quantity. Same as `new` after calling the method.
    """
    if mixing == 1.0:
        return new

    if isinstance(old, Gf) and isinstance(new, Gf):
        new << new * mixing + old * (1 - mixing)
    elif isinstance(old, BlockGf) and isinstance(new, BlockGf):
        for name in old.indices:
            new[name] << new[name] * mixing + old[name] * (1 - mixing)
    else:
        raise ValueError("Both `new` and `old` must be either Gf or BlockGf objects.")
    return new
