import torch

from fenics import *
from fenics_adjoint import *

from torchfenics.numpy_fenics import numpy_to_fenics, fenics_to_numpy


from abc import ABC, abstractmethod

class ForwardModel(ABC):
    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def input_templates(self):
        pass


class FEniCSFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, forward_model, *args):
        numpy_vars = []
        for x, t in zip(args, forward_model.input_templates()):
            if torch.is_tensor(x):
                x = x.detach().numpy()
            numpy_vars.append(numpy_to_fenics(x, t))

        tape = Tape()
        set_working_tape(tape)
        fenics_var = forward_model.forward(*numpy_vars)

        ctx.tape = tape
        ctx.numpy_vars = numpy_vars
        ctx.fenics_var = fenics_var

        return torch.from_numpy(fenics_to_numpy(fenics_var))

    @staticmethod
    def backward(ctx, grad_output):
        adj_input = numpy_to_fenics(grad_output.numpy(), ctx.fenics_var)

        if isinstance(adj_input, Function):
            adj_input = adj_input.vector()

        grads = compute_gradient(ctx.fenics_var, [Control(x) for x in ctx.numpy_vars], tape=ctx.tape, adj_input=adj_input)
        grads = [torch.from_numpy(fenics_to_numpy(grad)) for grad in grads]
        return tuple([None] + grads)


class FEniCS(torch.nn.Module):
    def __init__(self, forward_model):
        super(FEniCS, self).__init__()
        self.forward_model = forward_model

    def forward(self, *args):
        out = [FEniCSFunction.apply(self.forward_model, *x) for x in zip(*args)]
        return torch.stack(out)

