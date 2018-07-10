import torch

from fenics import *
from fenics_adjoint import *

from torchfenics.torch_fenics import FEniCS, ForwardModel


class Squares(ForwardModel):
    def __init__(self):
        mesh = IntervalMesh(4, 0, 1)
        self.V = FunctionSpace(mesh, 'DG', 0)

    def forward(self, f1, f2):
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        a = u * v * dx
        L = f1**2 * f2**2 * v * dx

        u_ = Function(self.V)
        solve(a == L, u_)

        return u_

    def input_templates(self):
        return [Function(self.V), Function(self.V)]


class Poisson(ForwardModel):
    def __init__(self):
        mesh = UnitSquareMesh(10, 10)
        self.V = FunctionSpace(mesh, 'P', 1)

    def forward(self, f):
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        a = inner(grad(u), grad(v)) * dx
        L = f * v * dx

        bc = DirichletBC(self.V, 0, 'on_boundary')

        u_ = Function(self.V)
        solve(a == L, u_, bc)

        return u_

    def input_templates(self):
        return [Constant(0)]



def test_squares():
    x1 = torch.autograd.Variable(torch.tensor([[1, 2, 3, 4],
                                               [2, 3, 5, 6]]).double(), requires_grad=True)
    x2 = torch.autograd.Variable(torch.tensor([[2, 3, 5, 6],
                                               [1, 2, 2, 1]]).double(), requires_grad=True)
    fenics = FEniCS(Squares())
    assert torch.autograd.gradcheck(fenics, (x1, x2))


def test_poisson():
    f = torch.tensor([[1]])
    fenics = FEniCS(Poisson())
    assert torch.autograd.gradcheck(fenics, (f))



if __name__ == '__main__':
    x1 = torch.autograd.Variable(torch.tensor([[1, 2, 3, 4],
                                               [2, 3, 5, 6]]).double(), requires_grad=True)
    x2 = torch.autograd.Variable(torch.tensor([[2, 3, 5, 6],
                                               [1, 2, 2, 1]]).double(), requires_grad=True)
    fenics = FEniCS(Squares())
    print(fenics(x1, x2))
