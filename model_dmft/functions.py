# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2024-07-26

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import gftool as gt
import numpy as np
import numpy.typing as npt
from triqs.gf import Gf, MeshImFreq, MeshReFreq
from triqs.gf.descriptors import Base

Args = Optional[Tuple[Any, ...]]
Kwargs = Optional[Dict[str, Any]]
Func = Callable[[Gf], Gf]

Onsite = Union[float, Sequence[float], Gf]


# noinspection PyPep8Naming
class PartialFunction(Base):
    """Partial function implementation of TRIQS Gf Function descriptor."""

    def __init__(self, function: Func, *args, **kwargs):
        if not callable(function):
            raise RuntimeError("function must be callable!")
        super().__init__(function=function, args=args, kwargs=kwargs)

    # noinspection PyUnresolvedReferences
    def __call__(self, G: Gf) -> Gf:
        """Evaluate the function on the mesh and store the result in the Gf."""
        return self.function(G, *self.args, **self.kwargs)


# noinspection PyPep8Naming
class Partial(ABC):
    """Partial function hanlder for TRIQS Gf Function descriptors.

    This class is used to create partial functions that can be applied to Gfs.
    It acts like a proxy for the actual function in order to be able to pass arguments
    to `triqs.gf.descriptor_base.Function` objects.
    """

    @abstractmethod
    def function(self, G: Gf, **kwargs) -> Gf:
        """Function definition. Must be implemented by the user."""
        pass

    def __call__(self, **kwargs) -> PartialFunction:
        """Return a partial function that can be applied to Gfs."""
        return PartialFunction(self.function, **kwargs)


# noinspection PyPep8Naming
class Ht(Partial, ABC):
    """Base class for partial lattice Hilbert transforms using TRIQS Functions.

    Supports an on-site energy, chemical potewntial and a (diagonal) self energy.

    Parameters
    ----------
    half_bandwidth : float, optional
        Half bandwidth of the lattice, by default 1.
    eps : float, optional
        Default on-site energy. Can be overwritten in the transform call.
    mu : float, optional
        Baseline chemical potential, by default 0.0. Is added to the value passed to the
        transform call
    """

    def __init__(self, half_bandwidth: float = 1.0, eps: Onsite = 0.0, mu: float = 0.0):
        self.half_bandwidth = half_bandwidth
        self.eps = eps
        self.mu = mu

    @abstractmethod
    def transform(self, G: Gf, Sigma: Gf, eps: Onsite, mu: float, eta: float) -> Gf:
        """Transform the Green's function G.

        Has to be implemented by the deriving class.

        Parameters
        ----------
        G : Gf
            The Green's function to transform.
        Sigma : Gf
            The diagonal self energy to use in the transformation.
        eps : float, optional
            The on-site energy. Shifts the Green's function.
        mu : float, optional
            Chemical potential. Sets the Fermi level.
        eta : float, optional
            Imaginary broadening.

        Returns
        -------
        Gf
            The transformed Green's function.
        """
        pass

    def function(self, G: Gf, Sigma: Gf = None, eps: Onsite = None, mu: float = None, eta: float = 0.0) -> Gf:
        """Function definition for PartialFunction, see `transform`."""
        if Sigma is None:
            Sigma = G.copy()
            Sigma.zero()
        eps = self.eps if eps is None else eps
        mu = self.mu + mu if mu is not None else self.mu
        return self.transform(G, Sigma, eps, mu, eta)

    def __call__(self, Sigma: Gf = None, eps: Onsite = None, mu: float = None, eta: float = 0.0) -> PartialFunction:
        """Return a partial function of `tansform` that can be applied to Gfs.

        Sigma : Gf, optional
            Diagonal self energy to use in the transformation. By default, no
            self energy is used in the transformation.
        eps : float, optional
            On-site energy. By default, the default on-site energy `self.eps` is used.
        mu : float, optional
            Additional chemical potential. If passed, the value is added to the
            base-line chemical potential `self.eps`.
        eta : float, optional
            Imaginary broadening, by default 0.
        """
        return super().__call__(Sigma=Sigma, eps=eps, mu=mu, eta=eta)


# noinspection PyPep8Naming
class SemiCircularHt(Ht):
    r"""Hilbert transform of a semicircular density of states with self energy, i.e.

     .. math::
        g(z - ε - Σ) = \int \frac{A(\omega)}{z - ε - Σ - \omega} d\omega

    where :math:`A(\omega) = \theta( D - |\omega|) 2 \sqrt{ D^2 - \omega^2}/(\pi D^2)`.

    (Only works in combination with frequency Green's functions.)
    """

    def transform(self, G: Gf, Sigma: Gf, eps: Onsite, mu: float, eta: float) -> Gf:
        ndim = 0 if len(G.target_shape) == 0 else G.target_shape[0]
        eye = complex(1, 0) if ndim == 0 else np.identity(ndim, np.complex128)
        x = np.array(list(G.mesh.values()))
        if ndim > 0:
            x = x[:, None, None]

        om = x + mu - eps - Sigma.data
        D = self.half_bandwidth
        D2 = D**2

        if isinstance(G.mesh, MeshImFreq):
            sign = np.copysign(1, om.imag)
            G.data[...] = (om - 1j * sign * np.sqrt(D2 - om**2)) / D2 * 2 * eye

        elif isinstance(G.mesh, MeshReFreq):
            band = (-D < om.real) & (om.real < D)
            f = 2 / D2
            sign = np.copysign(1, om[~band].real)
            G.data[band, ...] = f * (om[band] - 1j * np.sqrt(D2 - om[band] ** 2))
            G.data[~band, ...] = f * (om[~band] - sign * np.sqrt(om[~band] ** 2 - D2))

        else:
            raise TypeError("This HilbertTransform is only correct in frequency")

        return G


# noinspection PyPep8Naming
class HilbertTransform(Ht):
    """General Hilbert transform Function descriptor using `gftool` methods.

    Parameters
    ----------
    name : {bethe, square, triangular, honeycomb} str
        Name of the lattice Green's function to use.
    half_bandwidth : float, optional
        Half bandwidth of the lattice, by default 2.
    mu : float, optional
        Baseline chemical potential, by default 0.0.
    eps : float, optional
        Default on-site energy. Can be overwritten in the transform call.
    """

    def __init__(self, name: str, half_bandwidth: float = 2.0, mu: float = 0.0, eps: Onsite = 0.0):
        try:
            self.func = getattr(gt.lattice, name).gf_z
        except AttributeError:
            raise ValueError(f"Unknown lattice Green's function: {name}")
        super().__init__(half_bandwidth, eps, mu)

    @property
    def partial_func(self) -> Callable[[npt.ArrayLike], npt.ArrayLike]:
        return partial(self.func, half_bandwidth=self.half_bandwidth)

    def transform(self, G: Gf, Sigma: Gf, eps: Onsite, mu: float, eta: float) -> Gf:
        if isinstance(eps, Gf):
            eps = eps.data
        ndim = 0 if len(G.target_shape) == 0 else G.target_shape[0]
        eye = complex(1, 0) if ndim == 0 else np.identity(ndim, np.complex128)
        x = np.array(list(G.mesh.values())) + 1j * eta
        if ndim > 0:
            x = x[:, None, None]
        om = x + mu - eps - Sigma.data

        if isinstance(G.mesh, MeshImFreq):
            data = self.func(om, half_bandwidth=self.half_bandwidth)
            G.data[...] = data * eye
        elif isinstance(G.mesh, MeshReFreq):
            data = self.func(om, half_bandwidth=self.half_bandwidth)
            if eta == 0.0:
                data.imag *= np.copysign(1, om.real)
            G.data[...] = data
        else:
            raise TypeError("This HilbertTransform is only correct in frequency")

        return G
