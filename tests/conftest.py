import pytest
import sys
sys.path.append(sys.path[0]+'\\..')

from modules.trisolver import TriSolver

@pytest.fixture(scope="module")
def solver():
    return TriSolver('./assets/regular_tri_grid64.obj')