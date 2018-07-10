from fenics import *
from fenics_adjoint import *

def pytest_runtest_setup(item):
    """ Hook function which is called before every test """
    set_log_active(False)
