import fenics
fenics.set_log_level(fenics.LogLevel.ERROR)

from .torch_fenics import FEniCSModel
from .torch_fenics import FEniCSModule
from .torch_fenics import FEniCSFunction

from .numpy_fenics import fenics_to_numpy
from .numpy_fenics import numpy_to_fenics
