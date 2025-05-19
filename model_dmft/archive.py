# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-05-19

from pathlib import Path
from typing import List, Union

import h5

from .input import InputParameters
from .utility import GfLike


class Archive(h5.HDFArchive):
    def __init__(self, file: Union[str, Path], mode: str = "r", **kwargs):
        super().__init__(str(file), mode, **kwargs)

    def get_quantities(self) -> List[str]:
        keys = filter(lambda x: "-" not in x, self.keys())
        return list(keys)

    def get_iteration(self) -> int:
        return self["it"]

    def get_params(self) -> InputParameters:
        return self["params"]

    def get_g_coh(self, it: int = None) -> GfLike:
        return self[f"g_coh-{it}"] if it is not None else self["g_coh"]

    def get_g_cmpt(self, it: int = None) -> GfLike:
        return self[f"g_cmpt-{it}"] if it is not None else self["g_cmpt"]

    def get_g_l(self, it: int = None) -> GfLike:
        return self[f"g_l-{it}"] if it is not None else self["g_l"]

    def get_g_tau(self, it: int = None) -> GfLike:
        return self[f"g_tau-{it}"] if it is not None else self["g_tau"]

    def get_sigma_cpa(self, it: int = None) -> GfLike:
        return self[f"sigma_cpa-{it}"] if it is not None else self["sigma_cpa"]

    def get_sigma_dmft(self, it: int = None) -> GfLike:
        return self[f"sigma_dmft-{it}"] if it is not None else self["sigma_dmft"]

    def get_sigma_dmft_raw(self, it: int = None) -> GfLike:
        return self[f"sigma_dmft_raw-{it}"] if it is not None else self["sigma_dmft_raw"]

    def get_delta(self, it: int = None) -> GfLike:
        return self[f"delta-{it}"] if it is not None else self["delta"]

    def get_occ(self, it: int = None) -> GfLike:
        return self[f"occ-{it}"] if it is not None else self["occ"]

    def get_err_g(self, it: int = None) -> GfLike:
        return self[f"err_g-{it}"] if it is not None else self["err_g"]

    def get_err_sigma(self, it: int = None) -> GfLike:
        return self[f"err_sigma-{it}"] if it is not None else self["err_sigma"]

    def get_err_occ(self, it: int = None) -> GfLike:
        return self[f"err_occ-{it}"] if it is not None else self["err_occ"]
