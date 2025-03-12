# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2024-08-05

"""General utility functions for the package."""

import os
import warnings
from typing import Any, Generator, List, Tuple, Union

import numpy as np
from colorama import Fore, Style
from numpy.typing import DTypeLike
from triqs.gf import BlockGf, Gf, MeshImFreq, MeshImTime, MeshReFreq, MeshReTime
from triqs.utility import mpi

GfStruct = List[Tuple[str, int]]
GfLike = Union[Gf, BlockGf]
MeshLike = Union[MeshReFreq, MeshImFreq, MeshReTime, MeshImTime]

# Names for up/down - Allowed pairs are: 'up'/'down' or 'up'/'dn'
UP, DN = "up", "dn"

SIGMA = np.array([-0.5, +0.5])


COLORS = {
    "black": Fore.BLACK,
    "red": Fore.RED,
    "green": Fore.GREEN,
    "yellow": Fore.YELLOW,
    "blue": Fore.BLUE,
    "magenta": Fore.MAGENTA,
    "cyan": Fore.CYAN,
    "white": Fore.WHITE,
    "light_black": Fore.LIGHTBLACK_EX,
    "light_red": Fore.LIGHTRED_EX,
    "light_green": Fore.LIGHTGREEN_EX,
    "light_yellow": Fore.LIGHTYELLOW_EX,
    "light_blue": Fore.LIGHTBLUE_EX,
    "light_magenta": Fore.LIGHTMAGENTA_EX,
    "light_cyan": Fore.LIGHTCYAN_EX,
    "light_white": Fore.LIGHTWHITE_EX,
    "k": Fore.BLACK,
    "r": Fore.RED,
    "g": Fore.GREEN,
    "y": Fore.YELLOW,
    "b": Fore.BLUE,
    "m": Fore.MAGENTA,
    "c": Fore.CYAN,
    "w": Fore.WHITE,
    "lk": Fore.LIGHTBLACK_EX,
    "lr": Fore.LIGHTRED_EX,
    "lg": Fore.LIGHTGREEN_EX,
    "ly": Fore.LIGHTYELLOW_EX,
    "lb": Fore.LIGHTBLUE_EX,
    "lm": Fore.LIGHTMAGENTA_EX,
    "lc": Fore.LIGHTCYAN_EX,
    "lw": Fore.LIGHTWHITE_EX,
}


def style(text: Any, fg: str = "", bg: str = "", dim: bool = False) -> str:
    """Return a styled string."""
    text = str(text)
    fg = COLORS.get(fg, fg)
    bg = COLORS.get(bg, bg)
    s = fg + bg
    if dim:
        s += Style.DIM
    return fg + text + Style.RESET_ALL if s else text


def report(text: Any, once: bool = True) -> None:
    """Print a message to the console."""
    text = str(text)
    # text = style(text, fg=fg, bg=bg, dim=dim)
    if not once or mpi.is_master_node():
        mpi.report(text)


class WorkingDir:
    def __init__(self, path):
        self._prev = os.getcwd()
        self.path = path

    def __enter__(self):
        if self.path != self._prev:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            os.chdir(self.path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path != self._prev:
            os.chdir(self._prev)


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


def check_broadening(mesh: MeshLike, eta: float = 0.0) -> None:
    """Warn if the broadening seems to be invalid.

    The broadening parameter `η` should be used for real frequency meshes, but not for
    imaginary time meshes. If the broadening is smaller than the mesh discretization,
    the results also may be inaccurate.
    """
    if isinstance(mesh, MeshImFreq):
        if eta:
            warnings.warn(f"Broadening should not be used for imaginary meshes, got eta={eta}.")
    elif isinstance(mesh, MeshReFreq):
        if not eta:
            warnings.warn("No broadening is used for real frequency mesh!")
        else:
            w = np.array(list(mesh.values()))
            dw = min(np.diff(w))
            if eta < dw:
                warnings.warn(
                    f"Broadening {eta} is larger than mesh discretization {dw}. "
                    "This may lead to inaccurate results."
                )


def blockgf(
    mesh: MeshLike,
    names: List[str] = None,
    indices: List[int] = None,
    target_shape: List[int] = None,
    gf_struct: GfStruct = None,
    blocks: Union[GfLike, List[GfLike]] = None,
    copy: bool = False,
    name: str = "G",
) -> BlockGf:
    """Create an empty or filled BlockGf.

    Parameters
    ----------
    mesh : MeshReFreq or MeshImFreq
        The mesh for the Green's functions.
    names : List[str], optional
        The names of the Green's functions.
    indices : List[int], optional
        The indices defining the target shape of the Green's functions. Either `indices` or
        `target_shape` has to be provided if no blocks are given.
    target_shape : List[int], optional
        The target shape of the Green's functions. Either `indices` or `target_shape` has to be
         provided if no blocks are given.
    gf_struct : GfStruct, optional
        The structure of the Green's functions. If provided, `names` and `indices` and
        `target_shape` are ignored.
    blocks : Gf or BlockGf or List[Gf] or List[BlockGf], optional
        The blocks used to fill the BlockGf. If a single Gf or BlockGf is given is used for all
        blocks of the resulting BlockGf.
    copy : bool, optional
        If True, the blocks are copied, otherwise they are used as is. Only applies if
        all blocks are passed.
    name : str, optional
        The name of the BlockGf.

    Returns
    -------
    BlockGf
        The new BlockGf object.
    """
    if gf_struct is not None:
        names = [name for name, _ in gf_struct]
        norbs = np.unique([norbs for _, norbs in gf_struct])
        assert len(norbs) == 1, "All Gfs must have the same number of orbitals."
        indices = list(range(norbs[0]))
    else:
        if names is None:
            raise ValueError("Either gf_struct or names must be provided!")
        if blocks is None and target_shape is None and indices is None:
            raise ValueError("Either gf_struct, blocks, target_shape or indices must be provided!")

    if blocks is None:
        copy = True  # Force copy
        blocks = [Gf(mesh=mesh, indices=indices, target_shape=target_shape) for _ in names]
    elif isinstance(blocks, (Gf, BlockGf)):
        copy = True  # Force copy
        blocks = [blocks] * len(names)

    return BlockGf(name_list=names, block_list=blocks, make_copies=copy, name=name)


def toarray(obj: Union[MeshLike, GfLike], dtype: DTypeLike = None) -> np.ndarray:
    """Converts a supported TRIQS object to an array.

    Currently, TRIQS Mesh and Gf objects are supported.

    Parameters
    ----------
    obj : Union[MeshLike, GfLike]
        The object to convert to an array. If the object is a TRIQS mesh or Gf the data
        of the object will be returned directly as an array. If the object is a (nested)
        TRIQS BlockGf, the data of all leaf Gfs will be returned as an array with a shape
        corresponding to the block sturcture and the leaf shape.
        *Note*: All leafs of the (nested) BlockGf need to have the same shape.
    dtype : DTypeLike, optional
        Data type to use for creating the array. If no dtpye is given the original
        datatype of the TRIQS object ios used.

    Returns
    -------
    np.ndarray
        The TRIQS object as numpy array.
    """
    if isinstance(obj, (MeshReFreq, MeshImFreq, MeshImTime, MeshReTime)):
        data = list(obj.values())
    elif isinstance(obj, Gf):
        data = obj.data
    elif isinstance(obj, BlockGf):

        def _bgf2arr(g: BlockGf) -> list:
            """Recursively convert the BlockGf to an array."""
            _data = list()
            for k in g.indices:
                _data.append(_bgf2arr(g[k]) if isinstance(g[k], BlockGf) else g[k].data)
            return _data

        data = _bgf2arr(obj)

    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")

    return np.asarray(data, dtype=dtype)


def walk_block_leafs(gf: BlockGf) -> Generator[Gf, None, None]:
    """Walk through a (nested) BlockGf and yield all leaf Gfs."""
    queue = [gf]
    while queue:
        parent = queue.pop(0)
        if isinstance(parent, Gf):
            yield parent
        else:
            for k, g in parent:
                queue.append(g)


def walk_block_paths(gf: BlockGf) -> Generator[List[str], None, None]:
    """Walk through a (nested) BlockGf and yield the key paths to all leaf Gfs."""
    queue = [(gf, [])]
    while queue:
        parent, path = queue.pop(0)
        if isinstance(parent, Gf):
            yield path
        else:
            for k, g in parent:
                queue.append((g, path + [k]))


def gf_target_shape(gf: GfLike) -> Tuple[int, ...]:
    """Return the target shape of a Gf or (nested) BlockGf."""
    if isinstance(gf, Gf):
        return gf.target_shape
    else:
        shapes = set()
        for g in walk_block_leafs(gf):
            shapes.add(g.target_shape)
        if len(shapes) > 1:
            raise ValueError("Target shapes of the Gfs do not match!")
        return shapes.pop()


def gf_shape(gf: GfLike) -> Tuple[int, ...]:
    """Return the shape of a Gf or (nested) BlockGf.

    If a BlockGf is provided, the shape of the block children is checked and an error
    is raised if the shapes do not match.
    """
    if isinstance(gf, Gf):
        return gf.data.shape
    else:
        queue = [(gf, [])]
        shapes = set()
        while queue:
            parent, shape = queue.pop(0)
            if isinstance(parent, Gf):
                shapes.add(tuple(shape) + parent.data.shape)
            else:
                for k, g in parent:
                    queue.append((g, shape + [len(parent)]))

    if len(shapes) > 1:
        raise ValueError("Shapes of the blocks do not match!")
    return shapes.pop()


def fill_gf(gf: GfLike, data: np.ndarray) -> None:
    """Fill a Gf or (nested) BlockGf with data.

    Parameters
    ----------
    gf : Gf or BlockGf
        The Gf or BlockGf to be filled.
    data : np.ndarray
        The data to be filled into the Gf. The shape must match the shape of the Gf.

    """
    # Check if the data shape matches the Gf shape
    shape = gf_shape(gf)
    if len(data.shape) < len(shape):
        data = data[..., np.newaxis, np.newaxis]
    elif len(data.shape) > len(shape) and data.shape[-1] == 1:
        data = data[..., 0, 0]

    if shape[:-1] != data.shape[:-1]:
        raise ValueError(f"Shape {data.shape} does not match shape of Gf {shape}!")

    if isinstance(gf, Gf):
        gf.data[...] = data[...]
    else:

        def _fill_block(g: BlockGf, d: np.ndarray) -> None:
            for k, v in zip(g.indices, d):
                if isinstance(g[k], BlockGf):
                    _fill_block(g[k], v)
                else:
                    g[k].data[...] = v[...]

        _fill_block(gf, data)


def check_convergence(old: GfLike, new: GfLike, relative=False) -> float:
    if isinstance(old, Gf):
        error = float(np.linalg.norm(old.data - new.data))
        if relative:
            error /= np.linalg.norm(old.data)
    else:
        norms = list()
        max_norm = 0
        for keys in walk_block_paths(old):
            item_old = old
            item_new = new
            for key in keys:
                item_old = item_old[key]
                item_new = item_new[key]
            norms.append(np.linalg.norm(item_old.data - item_new.data))
            max_norm = max(max_norm, np.linalg.norm(item_old.data))
        error = max(norms)

        if relative:
            error /= max_norm
    return error


def difference(old: GfLike, new: GfLike) -> float:
    """Calculate the absolute difference between two Green's functions."""
    if isinstance(old, Gf):
        error = float(np.max(np.abs(old.data - new.data)))
    else:
        errors = list()
        for keys in walk_block_paths(old):
            item_old = old
            item_new = new
            for key in keys:
                item_old = item_old[key]
                item_new = item_new[key]
            errors.append(float(np.max(np.abs(item_old.data - item_new.data))))
        error = max(errors)
    return error


def symmetrize_gf(gf: BlockGf) -> BlockGf:
    """Symmetrize the spin components of a BlockGf."""
    up, dn = list(gf.indices)
    g = 0.5 * (gf[up] + gf[dn])
    gf[up] << g
    gf[dn] << g
    return gf


def mesh_to_array(mesh: Union[MeshReFreq, MeshImFreq]) -> np.ndarray:
    """Convert a mesh to a numpy array of real values."""
    if isinstance(mesh, MeshImFreq):
        return np.array([x.value.imag for x in mesh])
    elif isinstance(mesh, MeshReFreq):
        return np.array([x.value.real for x in mesh])
    else:
        raise ValueError(f"Unknown mesh type '{type(mesh)}'.")
