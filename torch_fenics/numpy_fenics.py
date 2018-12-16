from fenics import *
from fenics_adjoint import *

import numpy as np


def fenics_to_numpy(fenics_var):
    """Convert FEniCS variable to numpy array"""
    if isinstance(fenics_var, Constant):
        return fenics_var.values()

    if isinstance(fenics_var, Function):
        np_array = fenics_var.vector().get_local()
        n = fenics_var.function_space().num_sub_spaces()
        # Reshape if function is multi-component
        if n != 0:
            np_array = np.reshape(np_array, (len(np_array) // n, n))
        return np_array

    if isinstance(fenics_var, GenericVector):
        return fenics_var.get_local()

    if isinstance(fenics_var, AdjFloat):
        return np.array(float(fenics_var), dtype=np.float_)

    raise ValueError('Cannot convert ' + str(type(fenics_var)))


def numpy_to_fenics(numpy_array, fenics_var_template):
    """Convert numpy array to FEniCS variable"""
    if isinstance(fenics_var_template, Constant):
        if numpy_array.shape == (1,):
            return Constant(numpy_array[0])
        else:
            return Constant(numpy_array)

    if isinstance(fenics_var_template, Function):
        np_n_sub = numpy_array.shape[-1]
        np_dim = np.prod(numpy_array.shape)

        function_space = fenics_var_template.function_space()
        fenics_dim = function_space.dim()
        fenics_n_sub = function_space.num_sub_spaces()

        u = Function(function_space)
        if (fenics_n_sub != 0 and np_n_sub != fenics_n_sub) or np_dim != fenics_dim:
            err_msg = 'Cannot convert numpy array to Function:' \
                      ' Wrong shape {} vs {}'.format(numpy_array.shape, u.vector().get_local().shape)
            raise ValueError(err_msg)

        if numpy_array.dtype != np.float_:
            err_msg = 'The numpy array must be of type {}, ' \
                      'but got {}'.format(np.float_, numpy_array.dtype)
            raise ValueError(err_msg)

        u.vector().set_local(np.reshape(numpy_array, fenics_dim))
        return u

    if isinstance(fenics_var_template, AdjFloat):
        return AdjFloat(numpy_array)

    err_msg = 'Cannot convert numpy array to {}'.format(fenics_var_template)
    raise ValueError(err_msg)

