from distutils.core import setup

setup(name='torch-fenics',
      version='0.1',
      description='PyTorch-FEniCS interface',
      author='Patrik Barkman',
      author_email='barkm@kth.se',
      packages=['torch_fenics'],
      install_requires=['fenics==2018.1.0', 'dolfin_adjoint==2018.1.0', 'torch==0.4.1'],
      dependency_links=['git+https://bitbucket.org/barkm/pyadjoint.git@torch-fenics#egg=dolfin_adjoint-2018.1.0']
      )


