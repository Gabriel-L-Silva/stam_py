import warnings
from numpy import cos, sin
import numpy as np
import pytest

def plot(solver, error, name:str = 'error'):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig = plt.figure(num=name, figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.set_title('Erro')
    fig.suptitle(f'Norma infito do erro = {max(error):1e}',fontsize=20)
    surf = ax1.plot_trisurf(solver.mesh.points[:,0], solver.mesh.points[:,1], error, cmap=cm.coolwarm)
    fig.colorbar(surf, ax=ax1, fraction=0.1, pad=0.2)
    plt.show()

@pytest.mark.parametrize("problem, solution", [
    (lambda x, y: (-2*cos(x)*cos(y),-2*cos(x)*cos(y)), lambda x, y: 2*sin(x+y)),
    (lambda x, y: (x, -y), lambda x, y: 0),
    (lambda x, y: (cos(y), sin(x)), lambda x, y: 0),
    (lambda x, y: (cos(y)+x, sin(x)-y), lambda x, y: 0),
    (lambda x, y: (cos(x**2+y), -2*x*cos(x**2+y)), lambda x, y: 0)
])
def test_divergence(solver, problem, solution):    
    solver.vectors = np.asarray([problem(p[0],p[1]) for p in solver.mesh.points])
    div_sol = [solution(p[0],p[1]) for p in solver.mesh.points]

    div = np.asarray([solver.divergence(pid) for pid in range(solver.mesh.n_points)])

    error = abs(div-div_sol)
    
    try:
        assert max(error) <= 10e-3
    except:
        plot(solver, error, "divergence")
        assert max(error) <= 10e-3

def test_gradient(solver):
    def grad_problem(x,y):
        return -2*cos(x)*cos(y)

    def grad_solution(x,y):
        return 2*cos(y)*sin(x), 2*cos(x)*sin(y)

    x0 = np.asarray([grad_problem(p[0],p[1]) for p in solver.mesh.points])
    grad_sol = np.asarray([grad_solution(p[0],p[1]) for p in solver.mesh.points])

    grad = solver.gradient(x0)

    error = np.sum(abs(grad-grad_sol),axis=1)

    assert max(error) <= 10e-3

def test_poisson(solver):
    def poisson_problem(x,y):
        return -2*cos(x)*cos(y)
    def poisson_solution(x,y):
        return cos(x)*cos(y) - 1 

    b = [poisson_problem(p[0],p[1]) if id not in solver.mesh.boundary else 0 for id, p in enumerate(solver.mesh.points)]
    poisson_sol = [poisson_solution(p[0],p[1]) for p in solver.mesh.points]

    #applying dirichilet to find exact solution
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.w[0] = np.identity(solver.mesh.n_points)[0]
    b[0] = poisson_sol[0]

    lapl = solver.poisson_solver(b)
    error = abs(lapl-poisson_sol)
    
    assert max(error) <= 10e-3

def test_pressure_projection(solver):
    def poisson_problem(x,y):
        return -2*cos(x)*cos(y)
    
    solver.vectors[:] = poisson_problem(solver.mesh.points[:,0], solver.mesh.points[:,1])[:,None]
    solver.computePressure(1/60.0)
    div = np.array([solver.divergence(pid) if pid not in solver.mesh.boundary else 0.0 for pid in range(solver.mesh.n_points)])

    error = abs(div)

    assert max(error) <= 10e-3
