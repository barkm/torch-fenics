
from fenics import *
set_log_level(LogLevel.ERROR)

from .torch_fenics import FEniCSModel
from .torch_fenics import FEniCSModule

from .numpy_fenics import fenics_to_numpy
from .numpy_fenics import numpy_to_fenics
