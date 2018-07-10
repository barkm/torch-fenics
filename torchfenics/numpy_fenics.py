from fenics import *
from fenics_adjoint import *

import numpy as np

def fenics_to_numpy(fenics_var):
    """ Convert FEniCS variable to NumpyArray """
    if isinstance(fenics_var, Function):
        return fenics_var.vector().get_local()
    elif isinstance(fenics_var, Constant):
        return fenics_var.values()
    elif isinstance(fenics_var, GenericVector):
        return fenics_var.get_local()
    elif isinstance(fenics_var, AdjFloat):
        return np.array(float(fenics_var), dtype=np.float_)
    else:
        raise ValueError('Cannot convert ' + str(type(fenics_var)))


def numpy_to_fenics(numpy_array, fenics_var_template):
    """ Convert NumpyArray to FEniCS variable """
    if isinstance(fenics_var_template, Constant):
        if numpy_array.shape == (1,):
            return Constant(numpy_array[0])
        else:
            return Constant(numpy_array)
    elif isinstance(fenics_var_template, Function):
        u = Function(fenics_var_template.function_space())
        if numpy_array.shape != u.vector().get_local().shape:
            raise ValueError('Cannot convert NumpyArray to Function: Wrong shape ' +
                             str(numpy_array.shape) + ' vs ' + str(u.vector().get_local().shape))
        if numpy_array.dtype != np.float_:
            raise ValueError('Wrong type: ' + str(numpy_array.dtype))
        u.vector().set_local(numpy_array)
        return u
    elif isinstance(fenics_var_template, AdjFloat):
        return AdjFloat(numpy_array)
    else:
        raise ValueError('Cannot convert')

