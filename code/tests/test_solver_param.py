import warnings
from numpy import cos, sin
import numpy as np
import pytest
import sys
sys.path.append(sys.path[0]+'\\..')
from modules.trisolver import TriSolver

@pytest.mark.parametrize("k", range(6,50,2))
@pytest.mark.parametrize("s", [9])
@pytest.mark.parametrize("d", [2])
@pytest.mark.parametrize("only_knn", [True])
def test_div_param(plot, k, s, d, only_knn):
    solver = TriSolver('./assets/regular_tri_grid64.obj', k, s, d, only_knn)

    problem, solution = lambda x, y: (y**2*(y*cos(x)**2*cos(y*cos(x)**2) + 3*sin(y*cos(x)**2)), y**4*sin(2*x)*cos(y*cos(x)**2)), lambda x, y: 0

    solver.vectors = np.asarray([problem(p[0],p[1]) for p in solver.mesh.points])
    div = np.asarray([solver.divergence(pid) if pid not in solver.mesh.boundary else 0.0 for pid in range(solver.mesh.n_points)])
    div_sol = [solution(p[0],p[1]) if id not in solver.mesh.boundary else 0.0 for id, p in enumerate(solver.mesh.points) ]

    error = abs(div-div_sol)

    plot.error_pos.append(error)
    plot.k.append(k)
    plot.values.append(max(error))
    try:
        assert max(error) <= 10e-3
    except:
        if max(error) <= 10e-2:
            warnings.warn(f"Precision not optimal: {max(error):1e}")
        # plot(solver, error, "divergence")
        assert max(error) <= 10e-2
    

@pytest.mark.parametrize("k", range(6,50,2))
@pytest.mark.parametrize("s", [9])
@pytest.mark.parametrize("d", [2])
@pytest.mark.parametrize("only_knn", [True])
def test_poisson_param(plot, k, s, d, only_knn):
    solver = TriSolver('./assets/regular_tri_grid64.obj', k, s, d, only_knn)

    problem, solution = lambda x, y: -2*cos(x)*cos(y), lambda x, y: cos(x)*cos(y) - 1

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
    
    plot.error_pos.append(error)
    plot.k.append(k)
    plot.values.append(max(error))
    try:
        assert max(error) <= 10e-3
    except:
        if max(error) <= 10e-2:
            warnings.warn(f"Precision not optimal: {max(error):1e}")
        # plot(solver, error, "divergence")
        assert max(error) <= 10e-2

@pytest.mark.parametrize("k", range(6,50,2))
@pytest.mark.parametrize("s", [9])
@pytest.mark.parametrize("d", [2])
@pytest.mark.parametrize("only_knn", [True])
def test_grad_param(plot, k, s, d, only_knn):
    solver = TriSolver('./assets/regular_tri_grid64.obj', k, s, d, only_knn)

    problem, solution = lambda x, y: -2*cos(x)*cos(y), lambda x, y: (2*cos(y)*sin(x), 2*cos(x)*sin(y))

    x0 = np.asarray([problem(p[0],p[1]) for p in solver.mesh.points])
    grad_sol = np.asarray([solution(p[0],p[1]) for p in solver.mesh.points])

    grad = solver.gradient(x0)

    error = np.sum(abs(grad-grad_sol),axis=1)

    plot.error_pos.append(error)
    plot.k.append(k)
    plot.values.append(max(error))
    try:
        assert max(error) <= 10e-3
    except:
        if max(error) <= 10e-2:
            warnings.warn(f"Precision not optimal: {max(error):1e}")
        # plot(solver, error, "divergence")
        assert max(error) <= 10e-2