# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2024-08-05

from .input import SolverParams, FtpsSolverParams, InputParameters
from .folder import walkdirs, Folder

try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover
    __version__ = "unknown"
