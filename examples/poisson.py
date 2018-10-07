import torch

# Import fenics and override necessary data structures with fenics_adjoint
from fenics import *
from fenics_adjoint import *

from torch_fenics import FEniCSModel, FEniCSModule


# Declare the FEniCS model corresponding to solving the Poisson equation
# with variable source term and boundary value
class Poisson(FEniCSModel):
    # Construct variables which can be reused for each forward pass in the constructor
    def __init__(self):
        # Call super constructor
        super(Poisson, self).__init__()

        # Create function space
        mesh = UnitIntervalMesh(20)
        self.V = FunctionSpace(mesh, 'P', 1)

        # Create trial and test functions
        u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        # Construct bilinear form
        self.a = inner(grad(u), grad(self.v)) * dx

    def forward(self, f, g):
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
        return [Constant(0),  # source term
                Constant(0),  # boundary value
                ]


if __name__ == '__main__':
    # Instantiate the FEniCS model
    poisson = Poisson()

    # Create the PyTorch module
    poisson_module = FEniCSModule(poisson)

    # Create N sets of input
    N = 10
    f = torch.rand(N, 1, requires_grad=True, dtype=torch.float64)
    g = torch.rand(N, 1, requires_grad=True, dtype=torch.float64)

    # Solve the Poisson equation N times
    u = poisson_module(f, g)

    # Construct functional
    J = u.sum()

    # Execute the backward pass
    J.backward()

    # Extract gradients
    dJdf = f.grad
    dJdg = g.grad
