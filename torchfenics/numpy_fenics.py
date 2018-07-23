from fenics import *
from fenics_adjoint import *

import numpy as np

def fenics_to_numpy(fenics_var):
    """ Convert FEniCS variable to NumpyArray """
    if isinstance(fenics_var, Constant):
        return fenics_var.values()
    elif isinstance(fenics_var, Function):
        np_array = fenics_var.vector().get_local()
        n = fenics_var.function_space().num_sub_spaces()
        if n != 0:
            np_array = np.reshape(np_array, (len(np_array) // n, n))
        return np_array
    elif isinstance(fenics_var, GenericVector):
        ret = fenics_var.get_local()
        return ret
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
        np_n_sub = numpy_array.shape[-1]
        np_dim = np.prod(numpy_array.shape)
        function_space = fenics_var_template.function_space()
        fenics_dim = function_space.dim()
        fenics_n_sub = function_space.num_sub_spaces()
        if np_n_sub != fenics_n_sub and np_dim != fenics_dim:
            raise ValueError('Cannot convert NumpyArray to Function: Wrong shape ' +
                             str(numpy_array.shape) + ' vs ' + str(fenics_dim // fenics_n_sub) +
                             ', ' + str(fenics_n_sub))
        if numpy_array.dtype != np.float_:
            raise ValueError('Wrong type: ' + str(numpy_array.dtype))
        u = Function(fenics_var_template.function_space())
        u.vector().set_local(np.reshape(numpy_array, fenics_dim))
        return u
    elif isinstance(fenics_var_template, AdjFloat):
        return AdjFloat(numpy_array)
    else:
        raise ValueError('Cannot convert')

