import torch

# Import fenics and override necessary data structures with fenics_adjoint
from fenics import *
from fenics_adjoint import *

import torch_fenics

# Declare the FEniCS model corresponding to solving the Poisson equation
# with variable source term and boundary value
class Poisson(torch_fenics.FEniCSModule):
    # Construct variables which can be reused in the constructor
    def __init__(self):
        # Call super constructor
        super().__init__()

        # Create function space
        mesh = UnitIntervalMesh(20)
        self.V = FunctionSpace(mesh, 'P', 1)

        # Create trial and test functions
        u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        # Construct bilinear form
        self.a = inner(grad(u), grad(self.v)) * dx

    def solve(self, f, g):
        # Construct linear form
        L = f * self.v * dx

        # Construct boundary condition
        bc = DirichletBC(self.V, g, 'on_boundary')

        # Solve the Poisson equation
        u = Function(self.V)
        solve(self.a == L, u, bc)

        # Return the solution
        return u

    def input_templates(self):
        # Declare templates for the inputs to Poisson.forward
        return Constant(0), Constant(0)


if __name__ == '__main__':
    # Construct the FEniCS model
    poisson = Poisson()

    # Create N sets of input
    N = 10
    f = torch.rand(N, 1, requires_grad=True, dtype=torch.float64)
    g = torch.rand(N, 1, requires_grad=True, dtype=torch.float64)

    # Solve the Poisson equation N times
    u = poisson(f, g)

    # Construct functional
    J = u.sum()

    # Execute the backward pass
    J.backward()

    # Extract gradients
    dJdf = f.grad
    dJdg = g.grad
