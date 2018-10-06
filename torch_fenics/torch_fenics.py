from abc import ABC, abstractmethod

from fenics import *
from fenics_adjoint import *
import torch
import numpy as np

from torch_fenics.numpy_fenics import numpy_to_fenics, fenics_to_numpy


class FEniCSModel(ABC):
    """Defines a model in FEniCS"""

    def __init__(self):
        super(FEniCSModel, self).__init__()
        self._fenics_input_templates = None
        self._numpy_input_templates = None

    @abstractmethod
    def input_templates(self):
        """Returns list of FEniCS variables to be used as input templates to FEniCSModel.compute"""
        pass

    @abstractmethod
    def forward(self, *args):
        """Run the forward pass of the FEniCSModel

        Input:
            args (tuple): FEniCS variables of same type an order as specified by ForwardModel.input_templates

        Output:
            outputs (tuple): results from the forward pass
        """
        pass

    def fenics_input_templates(self):
        if self._fenics_input_templates is None:
            self._fenics_input_templates = self.input_templates()
        return self._fenics_input_templates

    def numpy_input_templates(self):
        if self._numpy_input_templates is None:
            self._numpy_input_templates = [fenics_to_numpy(temp) for temp in self.fenics_input_templates()]
        return self._numpy_input_templates


class FEniCSFunction(torch.autograd.Function):
    """Wraps a ForwardModel as a PyTorch function"""

    @staticmethod
    def forward(ctx, fenics_model, *args):
        """Computes the output of a FEniCS model and saves a corresponding tape

        Input:
            forward_model (ForwardModel): Defines the forward pass
            args (tuple): tensor representation of the input to the forward pass

        Output:
            tuple of outputs
        """
        # Check that the number of inputs arguments is correct
        n_args = len(args)
        expected_nargs = len(fenics_model.fenics_input_templates())
        if n_args != expected_nargs:
            err_msg = 'Wrong number of arguments to {}.' \
                      ' Expected {} got {}.'.format(type(fenics_model), expected_nargs, n_args)
            raise ValueError(err_msg)

        # Convert input tensors to corresponding FEniCS variables
        fenics_inputs = []
        for inp, template in zip(args, fenics_model.fenics_input_templates()):
            if torch.is_tensor(inp):
                inp = inp.detach().numpy()
            fenics_inputs.append(numpy_to_fenics(inp, template))

        # Create tape associated with this forward pass
        tape = Tape()
        set_working_tape(tape)

        # Execute forward pass
        fenics_outputs = fenics_model.forward(*fenics_inputs)

        # If single output
        if not isinstance(fenics_outputs, tuple):
            fenics_outputs = (fenics_outputs,)

        # Save variables to be used for backward pass
        ctx.tape = tape
        ctx.fenics_inputs = fenics_inputs
        ctx.fenics_outputs = fenics_outputs

        # Return tensor representation of outputs
        return tuple(torch.from_numpy(fenics_to_numpy(fenics_output)) for fenics_output in fenics_outputs)

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
            adj_value = numpy_to_fenics(grad_output.numpy(), fenics_output)
            # Special case
            if isinstance(adj_value, Function):
                adj_value = adj_value.vector()
            adj_values.append(adj_value)

        # Skip first input since this corresponds to the FEniCSModel
        needs_input_grad = ctx.needs_input_grad[1:]

        # Check which gradients need to be computed
        controls = []
        for needs_grad, fenics_input in zip(needs_input_grad, ctx.fenics_inputs):
            if needs_grad:
                controls.append(Control(fenics_input))

        # Compute and accumulate gradient for each output with respect to each input
        accumulated_grads = [None for _ in range(len(needs_input_grad))]
        for fenics_output, adj_value in zip(ctx.fenics_outputs, adj_values):
            fenics_grads = compute_gradient(fenics_output,
                                            controls,
                                            tape=ctx.tape,
                                            adj_value=adj_value)

            # Convert FEniCS gradients to tensor representation
            numpy_grads = []
            for fenics_grad in fenics_grads:
                if fenics_grad is None:
                    numpy_grads.append(None)
                else:
                    numpy_grads.append(torch.from_numpy(fenics_to_numpy(fenics_grad)))

            # Accumulate gradients
            i = 0
            for j, needs_grad in enumerate(needs_input_grad):
                if needs_grad:
                    if numpy_grads[i] is not None:
                        if accumulated_grads[j] is None:
                            accumulated_grads[j] = numpy_grads[i]
                        else:
                            accumulated_grads[j] += numpy_grads[i]
                    i += 1

        # Prepend None gradient corresponding to FEniCSModel input
        return tuple([None] + accumulated_grads)


class FEniCSModule(torch.nn.Module):
    """Wraps a FEniCSFunction in a PyTorch module"""

    def __init__(self, fenics_model):
        """Create the module

        Input:
            fenics_model (FEniCSModel): The FEniCS model to use in FEniCSFunction
        """
        super(FEniCSModule, self).__init__()
        self.fenics_model = fenics_model

    def forward(self, *args):
        """ Returns the output of the FEniCSModel for several inputs

        Input:
            args (tuple): List of tensor representations of the input to the FEniCSModel.
                          Each element in the list should be on the format
                          N x M_1 x M_2 x ... where N is the number of inputs and
                          M_1 x M2 ... are the dimensions of each input argument
        Output:
            output: Tensor representation of the output of the FEniCS model on the format
                    N x P_1 x P_2 x ... where N is the number of inputs and P_1 x P_2 x ...
                    are the dimensions of each output
        """
        # Check that the number of inputs is the same for each input argument
        if len(args) != 0:
            n = args[0].shape[0]
            for arg in args[1:]:
                if arg.shape[0] != n:
                    raise ValueError('Number of inputs must be the same for each input argument.')

        # Check that each input argument has correct dimensions
        for i, (arg, template) in enumerate(zip(args, self.fenics_model.numpy_input_templates())):
            arg_shape = tuple(arg.shape)[1:]
            if arg_shape != template.shape:
                err_msg = 'Expected input shape {} for input' \
                          ' {} but got {}.'.format(template.shape, i, arg_shape)
                raise ValueError(err_msg)

        # Check that the inputs are of double precision
        # TODO: Investigate if double precision requirement can be relaxed
        for i, arg in enumerate(args):
            if (isinstance(arg, np.ndarray) and arg.dtype != np.float64) or \
               (torch.is_tensor(arg) and arg.dtype != torch.float64):
                err_msg = 'All inputs must be type {},' \
                          ' but got {} for input {}.'.format(torch.float64, arg.dtype, i)
                raise TypeError(err_msg)

        # Run the FEniCS model on each set of inputs
        outs = [FEniCSFunction.apply(self.fenics_model, *inp) for inp in zip(*args)]

        # Rearrange by output index and stack over number of input sets
        outs = [torch.stack(out) for out in zip(*outs)]

        if len(outs) == 1:
            return outs[0]
        else:
            return outs

