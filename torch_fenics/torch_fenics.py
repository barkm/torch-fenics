from abc import ABC, abstractmethod

import fenics
import fenics_adjoint
import torch
import numpy as np

import torch_fenics.numpy_fenics as numpy_fenics


class FEniCSFunction(torch.autograd.Function):
    """Executes the solve function of a FEniCSModule"""

    @staticmethod
    def forward(ctx, fenics_solver, *args):
        """Computes the output of a FEniCS model and saves a corresponding gradient tape

        Input:
            fenics_solver (FEniCSSolver): FEniCSSolver to be executed during the forward pass
            args (tuple): tensor representation of the input to fenics_solver.forward

        Output:
            tensor representation of the output from fenics_solver.solve
        """
        # Check that the number of inputs arguments is correct
        n_args = len(args)
        expected_nargs = len(fenics_solver.fenics_input_templates())
        if n_args != expected_nargs:
            raise ValueError(f'Wrong number of arguments to {fenics_solver}.'
                             f' Expected {expected_nargs} got {n_args}.')

        # Check that each input argument has correct dimensions
        for i, (arg, template) in enumerate(zip(args, fenics_solver.numpy_input_templates())):
            if arg.shape != template.shape:
                raise ValueError(f'Expected input shape {template.shape} for input'
                                 f' {i} but got {arg.shape}.')

        # Check that the inputs are of double precision
        for i, arg in enumerate(args):
            if (isinstance(arg, np.ndarray) and arg.dtype != np.float64) or \
               (torch.is_tensor(arg) and arg.dtype != torch.float64):
                raise TypeError(f'All inputs must be type {torch.float64},'
                                f' but got {arg.dtype} for input {i}.')

        # Convert input tensors to corresponding FEniCS variables
        fenics_inputs = []
        for inp, template in zip(args, fenics_solver.fenics_input_templates()):
            if torch.is_tensor(inp):
                inp = inp.detach().numpy()
            fenics_inputs.append(numpy_fenics.numpy_to_fenics(inp, template))

        # Create tape associated with this forward pass
        tape = fenics_adjoint.Tape()
        fenics_adjoint.set_working_tape(tape)

        # Execute forward pass
        fenics_outputs = fenics_solver.solve(*fenics_inputs)

        # If single output
        if not isinstance(fenics_outputs, tuple):
            fenics_outputs = (fenics_outputs,)

        # Save variables to be used for backward pass
        ctx.tape = tape
        ctx.fenics_inputs = fenics_inputs
        ctx.fenics_outputs = fenics_outputs

        # Return tensor representation of outputs
        return tuple(torch.from_numpy(numpy_fenics.fenics_to_numpy(fenics_output)) for fenics_output in fenics_outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Computes the gradients of the output with respect to the input

        Input:
            ctx: Context used for storing information from the forward pass
            grad_output: gradient of the output from successive operations
        """
        # Convert gradient of output to a FEniCS variable
        adj_values = []
        for grad_output, fenics_output in zip(grad_outputs, ctx.fenics_outputs):
            adj_value = numpy_fenics.numpy_to_fenics(grad_output.numpy(), fenics_output)
            # Special case
            if isinstance(adj_value, (fenics.Function, fenics_adjoint.Function)):
                adj_value = adj_value.vector()
            adj_values.append(adj_value)

        # Check which gradients need to be computed
        controls = list(map(fenics_adjoint.Control,
                            (c for g, c in zip(ctx.needs_input_grad[1:], ctx.fenics_inputs) if g)))

        # Compute and accumulate gradient for each output with respect to each input
        accumulated_grads = [None] * len(controls)
        for fenics_output, adj_value in zip(ctx.fenics_outputs, adj_values):
            fenics_grads = fenics_adjoint.compute_gradient(fenics_output, controls,
                                                           tape=ctx.tape, adj_value=adj_value)

            # Convert FEniCS gradients to tensor representation
            numpy_grads = [g if g is None else torch.from_numpy(numpy_fenics.fenics_to_numpy(g)) for g in fenics_grads]
            for i, (acc_g, g) in enumerate(zip(accumulated_grads, numpy_grads)):
                if g is None:
                    continue
                if acc_g is None:
                    accumulated_grads[i] = g
                else:
                    accumulated_grads[i] += g

        # Insert None for not computed gradients
        acc_grad_iter = iter(accumulated_grads)
        return tuple(None if not g else next(acc_grad_iter) for g in ctx.needs_input_grad)


class FEniCSModule(ABC, torch.nn.Module):
    """Solves a PDE defined in FEniCS"""

    def __init__(self):
        super().__init__()
        self._fenics_input_templates = None
        self._numpy_input_templates = None

    @abstractmethod
    def input_templates(self):
        """Returns templates of the input to FEniCSModule.solve

        Not intended to be called by the user. Instead uses FEniCSModule.fenics_input_templates

        Output:
            FEniCS variable or tuple of FEniCS variables
        """
        pass

    @abstractmethod
    def solve(self, *args):
        """Solve PDE defined in FEniCS

        Input:
            args (tuple): FEniCS variables of same type as specified by FEniCSModule.input_templates

        Output:
            outputs (tuple): results from solving the PDE
        """
        pass

    def fenics_input_templates(self):
        """Returns tuple of FEniCS variables corresponding to input templates to FEniCSModule.solve"""
        if self._fenics_input_templates is None:
            templates = self.input_templates()
            if not isinstance(templates, tuple):
                templates = (templates,)
            self._fenics_input_templates = templates
        return self._fenics_input_templates

    def numpy_input_templates(self):
        """Returns tuple of numpy representations of the input templates to FEniCSModule.solve"""
        if self._numpy_input_templates is None:
            self._numpy_input_templates = [numpy_fenics.fenics_to_numpy(temp) for temp in self.fenics_input_templates()]
        return self._numpy_input_templates

    def forward(self, *args):
        """ Executes solve through FEniCSFunction for multiple inputs

        Input:
            args (tuple): List of tensor representations of the input to the FEniCSModel.
                          Each element in the tuple should be on the format
                          N x M_1 x M_2 x ... where N is the batch size and
                          M_1 x M_2 ... are the dimensions of the input argument
        Output:
            output: Tensor representations of the output from the FEniCSModel on the format
                    N x P_1 x P_2 x ... where N is the batch size and P_1 x P_2 x ...
                    are the dimensions of the output
        """
        # Check that the number of inputs is the same for each input argument
        if len(args) != 0:
            n = args[0].shape[0]
            for arg in args[1:]:
                if arg.shape[0] != n:
                    raise ValueError('Number of inputs must be the same for each input argument.')

        # Run the FEniCS model on each set of inputs
        outs = [FEniCSFunction.apply(self, *inp) for inp in zip(*args)]

        # Rearrange by output index and stack over number of input sets
        outs = tuple(torch.stack(out) for out in zip(*outs))

        if len(outs) == 1:
            return outs[0]
        else:
            return outs
