# Torch-FEniCS

The `torch_fenics` package enables models defined in [FEniCS](https://fenicsproject.org) to be used as modules in
 [PyTorch](https://pytorch.org/).

## Install

Install version `2018.1.0` of FEniCS and run

```bash
pip install --process-dependency-links git+https://github.com/pbarkm/torch-fenics.git@master
```

## Details

FEniCS objects are represented in PyTorch using their corresponding vector representation. For 
finite element functions this corresponds to their coefficient representation. 

The package relies on [`dolfin-adjoint`](http://www.dolfin-adjoint.org/en/latest/) in order for the FEniCS module to be compatible with the
automatic differentiation framework in PyTorch

## Usage

1. Define the FEniCS model by deriving from the `torch_fenics.FEniCSModel` class.

    1. Implement the `torch_fenics.FEniCSModel.forward` method such that it executes
     the desired forward pass defined in FEniCS.
    
    2. Implement the `torch_fenics.FEniCSModel.input_templates` method which defines 
    the input types to `torch_fenics.FEniCSModel.forward`.

2. Construct the PyTorch module by giving an instance of derived class
as input when constructing `torch_fenics.FEniCSModule`.

3. The FEniCS model can then be executed by giving the `torch_fenics.FEniCSModule` the
appropriate vector representations of the inputs defined in 
`torch_fenics.FEniCSModel.input_templates`.  Each input to `torch_fenics.FEniCSModule`
should be on the form `N x D_1 x D_2 ...` where `N` is the number of sets of inputs such that
`torch.FEniCSModel.forward` is executed `N` times.

## Example

The `torch_fenics` package can for example be used to define a PyTorch module which solves the Poisson
equation using FEniCS.

First the process of solving the Poisson equation is defined in FEniCS by deriving the `torch_fenics.FEniCSModel` class

```python
# Import fenics as well as fenics_adjoint
from fenics import *
from fenics_adjoint import *

from torch_fenics import FEniCSModel

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
```

Through `torch_fenics.FEniCSModule` this FEniCS model can be used as a PyTorch module

```python
from torch_fenics import FEniCSModule

# Construct the FEniCS model
poisson = Poisson()

# Create the PyTorch module
poisson_module = FEniCSModule(poisson)
```

The `Poisson.forward` function can now be executed by giving the PyTorch module 
the appropriate vector input corresponding to the input templates declared in 
`Poisson.input_templates`. In this case the vector representation of the 
template `Constant(0)` is simply a scalar. 

```python
import torch

# Create N sets of input
N = 10
f = torch.rand(N, 1, requires_grad=True, dtype=torch.float64)
g = torch.rand(N, 1, requires_grad=True, dtype=torch.float64)

# Solve the Poisson equation N times
u = poisson_module(f, g)
```

The output of the `torch_fenics.FEniCSModule` can now be used to construct some 
functional. Consider summing up the coefficients of the solutions to the Poisson
equation

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

Install FEniCS `2018.1.0` and the dependencies in [`requirements.txt`](requirements.txt)

```
pip install -r requirements.txt
```

The unit-tests can then be run as follows

```
python -m pytest tests
```
