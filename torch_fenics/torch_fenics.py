from abc import ABC, abstractmethod

from fenics import *
from fenics_adjoint import *
import torch

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
    def compute(self, *args):
        """Returns output from the FEniCSModel

        Input:
            args (tuple): FEniCS variables of same type an order as specified by ForwardModel.input_templates
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
        """

        # Convert input tensors to corresponding FEniCS variables
        numpy_inputs = []
        for inp, template in zip(args, fenics_model.fenics_input_templates()):
            if torch.is_tensor(inp):
                inp = inp.detach().numpy()
            numpy_inputs.append(numpy_to_fenics(inp, template))

        # Create tape associated with this forward pass
        tape = Tape()
        set_working_tape(tape)

        # Execute forward pass
        fenics_output = fenics_model.compute(*numpy_inputs)

        # Save variables to be used for backward pass
        ctx.tape = tape
        ctx.numpy_inputs = numpy_inputs
        ctx.fenics_output = fenics_output

        # Return tensor representation of output
        return torch.from_numpy(fenics_to_numpy(fenics_output))

    @staticmethod
    def backward(ctx, grad_output):
        """Computes the gradients of the output with respect to the input

        Input:
            ctx: Context used for storing information from the forward pass
            grad_output: gradient of the output from successive operations
        """
        # Convert gradient of output to a FEniCS variable
        adj_input = numpy_to_fenics(grad_output.numpy(), ctx.fenics_output)

        # Special case
        if isinstance(adj_input, Function):
            adj_input = adj_input.vector()

        # Skip first input since this corresponds to the FEniCSModel
        needs_input_grad = ctx.needs_input_grad[1:]

        # Check which gradients need to be computed
        controls = []
        for needs_grad, inp in zip(needs_input_grad, ctx.numpy_inputs):
            if needs_grad:
                controls.append(Control(inp))

        # Compute gradient with respect to each input
        grads = compute_gradient(ctx.fenics_output,
                                 controls,
                                 tape=ctx.tape,
                                 adj_input=adj_input)

        # Convert FEniCS gradients to tensor representation
        grads = [torch.from_numpy(fenics_to_numpy(grad)) for grad in grads]

        # Insert None for inputs that did not need gradient
        all_grads = [None]  # First input corresponds to the FEniCSModel
        i = 0
        for needs_grad in needs_input_grad:
            if needs_grad:
                all_grads.append(grads[i])
                i += 1
            else:
                all_grads.append(None)

        return tuple(all_grads)


class FEniCS(torch.nn.Module):
    """Wraps a FEniCSFunction in a PyTorch module"""

    def __init__(self, fenics_model):
        """Create the module

        Input:
            fenics_model (FEniCSModel): The FEniCS model to use in FEniCSFunction
        """
        super(FEniCS, self).__init__()
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
        # Check that the number of inputs arguments is correct
        n_args = len(args)
        expected_nargs = len(self.fenics_model.fenics_input_templates())
        if n_args != expected_nargs:
            err_msg = 'Wrong number of arguments to {}.' \
                      ' Expected {} got {}'.format(self.fenics_model, n_args, expected_nargs)
            raise ValueError(err_msg)

        # Check that the number of inputs is the same for each input argument
        if n_args != 0:
            n = args[0].shape[0]
            for arg in args[1:]:
                if arg.shape[0] != n:
                    raise ValueError('Number of inputs must be the same for each input argument')

        # Check that each input argument has correct dimensions
        for i, (arg, template) in enumerate(zip(args, self.fenics_model.numpy_input_templates())):
            arg_shape = tuple(arg.shape)[1:]
            if arg_shape != template.shape:
                err_msg = 'Expected input shape {} for input' \
                          ' {} but got {}'.format(template.shape, i, arg_shape)
                raise ValueError(err_msg)

        # Check that the inputs are of double precision
        for i, arg in enumerate(args):
            if arg.dtype != torch.float64:
                err_msg = 'All inputs must be type {},' \
                          ' but got {} for input {}'.format(torch.float64, arg.dtype, i)
                raise ValueError(err_msg)

        # Run the FEniCS model on each set of inputs
        out = [FEniCSFunction.apply(self.fenics_model, *inp) for inp in zip(*args)]

        # Stack output before returning
        return torch.stack(out)

