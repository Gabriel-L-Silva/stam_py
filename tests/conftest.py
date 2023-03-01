from matplotlib import animation
import numpy as np
import pytest
import sys
sys.path.append(sys.path[0]+'\\..')

from modules.trisolver import TriSolver

@pytest.fixture(scope="module")
def solver():
    return TriSolver('./assets/regular_tri_grid64.obj', 36, 9, 3, True)

class Plot:
    def __init__(self, solver) -> None:
        self.k = []
        self.values = []
        self.error_pos = []
        self.points = solver.mesh.points

@pytest.fixture(scope="module")
def plot(solver):
    p = Plot(solver)
    yield p
    import matplotlib.pyplot as plt
    from matplotlib import cm

    plt.plot(p.k, p.values,'-o')
    plt.show(block=True)

    def generate_triangular_surfaces(d):
        ax.clear()
        ax.set_xlim(0, np.pi)
        ax.set_ylim(0, np.pi)
        ax.set_title(f'k={2*d}')
        ax.plot_trisurf(p.points[:,0], p.points[:,1], abs(p.error_pos[d]), cmap=cm.coolwarm)
        
    # Attaching 3D axis to the figure
    my_dpi = 96
    fig = plt.figure(figsize=(720/my_dpi, 720/my_dpi), dpi=my_dpi)
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, np.pi)
    anim = animation.FuncAnimation(fig, generate_triangular_surfaces, len(p.error_pos), repeat=False)
    progress_callback = lambda i, n: print(f'Saving frame {i} of {n}')
    anim.save('k_error_anim.mp4', fps=1, progress_callback=progress_callback)
    