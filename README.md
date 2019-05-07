# Torch-FEniCS

The `torch-fenics` package enables models defined in [FEniCS](https://fenicsproject.org) to be used as modules in
 [PyTorch](https://pytorch.org/).

## Install

[Install FEniCS](https://fenicsproject.org/download/) and run

```bash
pip install git+https://github.com/pbarkm/torch-fenics.git@master
```

A clean install of the package and its dependencies can for example be done with [Conda](https://conda.io/docs/)

```bash
conda create --name torch-fenics
conda activate torch-fenics
conda install -c conda-forge fenics
pip install git+https://github.com/pbarkm/torch-fenics.git@master
```

## Details

FEniCS objects are represented in PyTorch using their corresponding vector representation. For 
finite element functions this corresponds to their coefficient representation. 

The package relies on [`dolfin-adjoint`](http://www.dolfin-adjoint.org/en/latest/) in order for the FEniCS module to be compatible with the
automatic differentiation framework in PyTorch

## Example

The `torch-fenics` package can for example be used to define a PyTorch module which solves the Poisson
equation using FEniCS.

The process of solving the Poisson equation in FEniCS can be specified as a PyTorch module by deriving the `torch_fenics.FEniCSModule` class

```python
# Import fenics and override necessary data structures with fenics_adjoint
from fenics import *
from fenics_adjoint import *

import torch_fenics

# Declare the FEniCS model corresponding to solving the Poisson equation
# with variable source term and boundary value
class Poisson(torch_fenics.FEniCSModule):
    # Construct variables which can be in the constructor
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
        # Declare templates for the inputs to Poisson.solve
        return Constant(0), Constant(0)
```

The `Poisson.solve` function can now be executed by giving the module 
the appropriate vector input corresponding to the input templates declared in 
`Poisson.input_templates`. In this case the vector representation of the 
template `Constant(0)` is simply a scalar. 

```python
# Construct the FEniCS model
poisson = Poisson()

# Create N sets of input
N = 10
f = torch.rand(N, 1, requires_grad=True, dtype=torch.float64)
g = torch.rand(N, 1, requires_grad=True, dtype=torch.float64)

# Solve the Poisson equation N times
u = poisson(f, g)
```

The output of the can now be used to construct some functional. Consider summing
up the coefficients of the solutions to the Poisson equation

```python
# Construct functional 
J = u.sum()
```

The derivative of this functional with respect to `f` and `g` can now be
computed using the `torch.autograd` framework.

```python
# Execute backward pass
J.backward() 

# Extract gradients
dJdf = f.grad
dJdg = g.grad
```

## Developing
Install dependencies

```bash
conda env create -n torch-fenics -f environment.yml
conda activate torch-fenics
```

Install package in editable mode

```python
pip install -e .[test]
```

The unit-tests can then be run as follows

```bash
python -m pytest tests
```
