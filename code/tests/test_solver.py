import warnings
from numpy import cos, log, sin
import numpy as np
import pytest

def plot(solver, error, name:str = 'error'):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    points = np.asarray([solver.mesh.points[pid] for pid in range(solver.mesh.n_points) if pid not in solver.mesh.boundary])
    fig = plt.figure(num=name, figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.set_title('Erro')
    fig.suptitle(f'Norma infito do erro = {max(error):1e}',fontsize=20)
    surf = ax1.plot_trisurf(points[:,0], points[:,1], error, cmap=cm.coolwarm)
    fig.colorbar(surf, ax=ax1, fraction=0.1, pad=0.2)
    plt.show()

@pytest.mark.parametrize("problem, solution", [
    pytest.param(lambda x, y: (-2*cos(x)*cos(y),-2*cos(x)*cos(y)), lambda x, y: 2*sin(x+y), id='{-2*cos(x)*cos(y),-2*cos(x)*cos(y)}'),
    pytest.param(lambda x, y: (x, -y), lambda x, y: 0, id='{x, -y}'),
    pytest.param(lambda x, y: (cos(y), sin(x)), lambda x, y: 0, id='{cos(y), sin(x)}'),
    pytest.param(lambda x, y: (cos(y)+x, sin(x)-y), lambda x, y: 0, id='{cos(y)+x, sin(x)-y'),
    pytest.param(lambda x, y: (cos(x**2+y), -2*x*cos(x**2+y)), lambda x, y: 0, id='{cos(x**2+y), -2*x*cos(x**2+y)}'),
    pytest.param(lambda x, y: (y**2*(y*cos(x)**2*cos(y*cos(x)**2) + 3*sin(y*cos(x)**2)), y**4*sin(2*x)*cos(y*cos(x)**2)), lambda x, y: 0, id='{y**2*(y*cos(x)**2*cos(y*cos(x)**2) + 3*sin(y*cos(x)**2)), y**4*sin(2*x)*cos(y*cos(x)**2)}')
])
def test_divergence(solver, problem, solution):    
    solver.vectors = np.asarray([problem(p[0],p[1]) for p in solver.mesh.points])

    div = np.asarray([solver.divergence(pid) for pid in range(solver.mesh.n_points) if pid not in solver.mesh.boundary])
    div_sol = [solution(p[0],p[1]) for id, p in enumerate(solver.mesh.points) if id not in solver.mesh.boundary]

    error = abs(div-div_sol)
    
    try:
        assert max(error) <= 10e-3
    except:
        warnings.warn("Precision not optimal")
        plot(solver, error, "divergence")
        assert max(error) <= 10e-1

@pytest.mark.parametrize("problem, solution", [
    pytest.param(lambda x, y: -2*cos(x)*cos(y), lambda x, y: (2*cos(y)*sin(x), 2*cos(x)*sin(y)), id='-2*cos(x)*cos(y)')
])
def test_gradient(solver, problem, solution):
    x0 = np.asarray([problem(p[0],p[1]) for p in solver.mesh.points])
    grad_sol = np.asarray([solution(p[0],p[1]) for p in solver.mesh.points])

    grad = solver.gradient(x0)

    error = np.sum(abs(grad-grad_sol),axis=1)

    assert max(error) <= 10e-3

@pytest.mark.parametrize("problem, solution", [
    pytest.param(lambda x, y: -2*cos(x)*cos(y), lambda x, y: cos(x)*cos(y) - 1, id='-2*cos(x)*cos(y)')
])
def test_poisson(solver, problem, solution):
    from scipy.sparse.linalg import spsolve

    b = [problem(p[0],p[1]) if id not in solver.mesh.boundary else 0 for id, p in enumerate(solver.mesh.points)]
    poisson_sol = [solution(p[0],p[1]) for p in solver.mesh.points]

    #applying dirichilet to find exact solution
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = solver.w.copy()
        w[0] = np.identity(solver.mesh.n_points)[0]
    b[0] = poisson_sol[0]

    lapl = spsolve(w, b)
    error = abs(lapl-poisson_sol)
    
    assert max(error) <= 10e-3

@pytest.mark.parametrize("problem", [
    pytest.param(lambda x, y: (x, -y), id='{x, -y}'),
    pytest.param(lambda x, y: (cos(y), sin(x)), id='{cos(y), sin(x)}'),
    pytest.param(lambda x, y: (cos(y)+x, sin(x)-y), id='{cos(y)+x, sin(x)-y'),
    pytest.param(lambda x, y: (cos(x**2+y), -2*x*cos(x**2+y)), id='{cos(x**2+y), -2*x*cos(x**2+y)}'),
    pytest.param(lambda x, y: (y**2*(y*cos(x)**2*cos(y*cos(x)**2) + 3*sin(y*cos(x)**2)), y**4*sin(2*x)*cos(y*cos(x)**2)), id='{y**2*(y*cos(x)**2*cos(y*cos(x)**2) + 3*sin(y*cos(x)**2)), y**4*sin(2*x)*cos(y*cos(x)**2)}'),
    pytest.param(lambda x, y: (y**2*log(x)*(3*sin(y*(x + y)) + y*(x + 2*y)*cos(y*(x + y))), -(y**3*(sin(y*(x + y)) + x*y*log(x)*cos(y*(x + y))))/x), id='y**2*log(x)*(3*sin(y*(x + y)) + y*(x + 2*y)*cos(y*(x + y))), -(y**3*(sin(y*(x + y)')
])
def test_pressure_projection(solver, problem):
    solver.vectors = np.asarray([problem(p[0],p[1]) for p in solver.mesh.points])
    
    #Compute Pressure (Act)
    x0 = solver.poisson_solver()

    ##Gradient
    grad = solver.gradient(x0)
    
    ##Apply pressure gradient
    solver.vectors -= grad
    
    ##Assert
    div = np.array([solver.divergence(pid) if pid not in solver.mesh.boundary else 0.0 for pid in range(solver.mesh.n_points)])
    error = abs(div)
    
    try:
        assert max(error) <= 10e-3
    except:
        warnings.warn("Precision not optimal")
        plot(solver, error, "pressure projection")
        assert max(error) <= 10e-1
