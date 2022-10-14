from math import pi
import time
import numpy as np
from tqdm import tqdm
try:
    from modules.tri_mesh import TriMesh
except:
    from tri_mesh import TriMesh
try:
    from modules.interpolator import Interpolator, RBFInterpolator
except:
    from interpolator import Interpolator, RBFInterpolator

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from numba import njit, prange

@njit(parallel=True)
def gradient(rbf, nring, lapl, n_points):
    grad = np.zeros((n_points,2))
    for pid in prange(n_points):        
        grad[pid] = (np.sum(rbf[pid][np.argwhere(nring[pid]).flatten(),1]*lapl[np.argwhere(nring[pid]).flatten()]),
                    np.sum(rbf[pid][np.argwhere(nring[pid]).flatten(),2]*lapl[np.argwhere(nring[pid]).flatten()]))
    return grad

@njit(parallel=True)
def divergent(rbf, nring, vectors, n_points, boundary=np.array([])):
    div = np.zeros(n_points)
    for pid in prange(n_points): 
        if pid not in boundary:
            div[pid]= np.sum(rbf[pid][np.argwhere(nring[pid]).flatten(),1]*vectors[np.argwhere(nring[pid]).flatten(),0] 
                            + rbf[pid][np.argwhere(nring[pid]).flatten(),2]*vectors[np.argwhere(nring[pid]).flatten(),1])
    return div

class TriSolver:
    def __init__(self, filename):
        self.mesh = TriMesh(filename)

        self.density = np.zeros((self.mesh.n_points))
        self.vectors = np.zeros((self.mesh.n_points,3))

        self.Interpolator = RBFInterpolator(self.mesh)

        self.init_poisson_weights()

        self.source_cells = set()

    def apply_boundary_condition(self, field=None):
        if type(field) == type(None):
            self.vectors[list(self.mesh.boundary),:2] = [0,0]
        else:
            field[list(self.mesh.boundary)] = 0
                    

    def computeExternalForces(self, dt):
        pass

    def computeViscosity(self, dt):
        pass

    def computePressure(self, dt):
        x0 = self.poisson_solver()
        grad = gradient(self.mesh.rbf, self.mesh.nring, x0, self.mesh.n_points)
        
        self.vectors[:,0] -= grad[:,0]
        self.vectors[:,1] -= grad[:,1]

        self.apply_boundary_condition()

    def computeAdvection(self, density, dt):
        new_pos = self.mesh.mesh.vertices - self.vectors*dt
        new_pos = np.clip(new_pos,0,pi)
        if density:
            self.density = np.clip(self.Interpolator(self.density, new_pos), 0, 1)
        else:
            self.vectors[:,0] = self.Interpolator(self.vectors[:,0], new_pos)
            self.vectors[:,1] = self.Interpolator(self.vectors[:,1], new_pos)

        self.apply_boundary_condition()

    def computeSource(self, dt):
        self.vectors[list(self.source_cells),:2] = [0,5]
        self.density[list(self.source_cells)] = 1

    def init_poisson_weights(self):
        data = []
        data_row = []
        data_col = []
        print('Building Poisson matrix...')
        for pid in tqdm(range(self.mesh.n_points)): 
            if pid in self.mesh.boundary:
                weights = (self.mesh.rbf[pid][np.argwhere(self.mesh.nring[pid]).flatten(),1]*self.mesh.normals[pid,0] 
                        + self.mesh.rbf[pid][np.argwhere(self.mesh.nring[pid]).flatten(),2]*self.mesh.normals[pid,1])
            else:
                weights = self.mesh.rbf[pid][np.argwhere(self.mesh.nring[pid]).flatten(),0]
            
            data[0:0] = weights
            data_row[0:0] = ([pid]*len(np.argwhere(self.mesh.nring[pid]).flatten()))
            data_col[0:0] = np.argwhere(self.mesh.nring[pid]).flatten()
        self.w = csr_matrix((data, (data_row, data_col)), 
                          shape = (self.mesh.n_points, self.mesh.n_points))

        
    def poisson_solver(self, testing=None):        
        if testing != None:
            b = testing
        else:
            b = divergent(self.mesh.rbf, self.mesh.nring, self.vectors[:,:2], self.mesh.n_points, np.array(list(self.mesh.boundary)))
        lapl = spsolve(self.w, b)
        return lapl

    def velocityStep(self, dt):
        self.computeExternalForces(dt)

        self.computeViscosity(dt)

        # self.computePressure(dt)

        self.computeAdvection(False, dt)

        self.computePressure(dt)

    def densityStep(self, dt):
        self.computeSource(dt)

        self.computeViscosity(dt)

        self.computeAdvection(True, dt)

    def update_fields(self, dt):
        self.apply_boundary_condition()
        self.velocityStep(dt)
        self.apply_boundary_condition()
        self.densityStep(dt)


def test_poisson():
    def poisson_problem(x,y):
        return -2*cos(x)*cos(y)
    def poisson_solution(x,y):
        return cos(x)*cos(y) - 1 
    
    solver = TriSolver('./assets/mesh8.obj')

    b = [poisson_problem(p[0],p[1]) if id not in solver.mesh.boundary else 0 for id, p in enumerate(solver.mesh.points)]
    poisson_sol = [poisson_solution(p[0],p[1]) for p in solver.mesh.points]

    #applying dirichilet to find exact solution
    solver.w[0] = np.identity(solver.mesh.n_points)[0]
    b[0] = poisson_sol[0]

    lapl = solver.poisson_solver(b)
    error = abs(lapl-poisson_sol)
    fig = plt.figure(num='Poisson',figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax1.set_title('Nossa solução')
    ax2.set_title('Solução Exata')
    ax3.set_title('Erro')
    # Plot the surface.
    surf = ax1.plot_trisurf(solver.mesh.points[:,0], solver.mesh.points[:,1], lapl, cmap=cm.coolwarm                        )
    surf = ax2.plot_trisurf(solver.mesh.points[:,0], solver.mesh.points[:,1], poisson_sol, cmap=cm.coolwarm                       )
    surf3 = ax3.plot_trisurf(solver.mesh.points[:,0], solver.mesh.points[:,1], error, cmap=cm.coolwarm                        )
    fig.colorbar(surf3, ax=ax3, fraction=0.1, pad=0.2)
    fig.suptitle(f'Norma infito do erro = {max(error):1e}', fontsize=20)
    return max(error)
    
def test_divergence():
    def div_problem(x,y):
        return -2*cos(x)*cos(y),-2*cos(x)*cos(y)

    def div_solution(x,y):
        return 2*sin(x+y)
    
    solver = TriSolver('./assets/mesh16.obj')

    solver.vectors = np.asarray([div_problem(p[0],p[1]) for p in solver.mesh.points])
    div_sol = [div_solution(p[0],p[1]) for p in solver.mesh.points]

    for times in range(50):
        start = time.time()
        #do some stuff
        div = divergent(solver.mesh.rbf, solver.mesh.nring, solver.vectors, solver.mesh.n_points, np.array(list(solver.mesh.boundary)))
        stop = time.time()
        duration = stop-start
        print(f'{times}:\t {duration}')

    error = abs(div-div_sol)
    fig = plt.figure(num='divergence', figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax1.set_title('Nossa solução')
    ax2.set_title('Solução Exata')
    ax3.set_title('Erro')
    fig.suptitle(f'Norma infito do erro = {max(error):1e}',fontsize=20)
    # Plot the surface.
    surf = ax1.plot_trisurf(solver.mesh.points[:,0], solver.mesh.points[:,1], div, cmap=cm.coolwarm                        )
    surf = ax2.plot_trisurf(solver.mesh.points[:,0], solver.mesh.points[:,1], div_sol, cmap=cm.coolwarm                       )
    surf3 = ax3.plot_trisurf(solver.mesh.points[:,0], solver.mesh.points[:,1], error, cmap=cm.coolwarm                        )
    fig.colorbar(surf3, ax=ax3, fraction=0.1, pad=0.2)
    return max(error)

def test_gradient():
    def grad_problem(x,y):
        return -2*cos(x)*cos(y)

    def grad_solution(x,y):
        return 2*cos(y)*sin(x), 2*cos(x)*sin(y)
    
    solver = TriSolver('./assets/mesh16.obj')

    x0 = np.asarray([grad_problem(p[0],p[1]) for p in solver.mesh.points])
    grad_sol = np.asarray([grad_solution(p[0],p[1]) for p in solver.mesh.points])

    grad = gradient(solver.mesh.rbf, solver.mesh.nring, x0, solver.mesh.n_points)

    error = np.sum(abs(grad-grad_sol),axis=1)
    fig = plt.figure(num='Gradient', figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax1.set_title('Nossa solução')
    ax2.set_title('Solução Exata')
    ax3.set_title('Erro')
    fig.suptitle(f'Norma infito do erro = {max(error):1e}',fontsize=20)
    # Plot the surface.
    surf1 = ax1.quiver(solver.mesh.points[:,0], solver.mesh.points[:,1], grad[:,0],grad[:,1])
    surf2 = ax2.quiver(solver.mesh.points[:,0], solver.mesh.points[:,1], grad_sol[:,0], grad_sol[:,1])
    surf3 = ax3.plot_trisurf(solver.mesh.points[:,0], solver.mesh.points[:,1], error, cmap=cm.coolwarm)
    fig.colorbar(surf3, ax=ax3, fraction=0.1, pad=0.2)
    return max(error)
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from math import cos, sin
    from matplotlib import cm
    #######################################################################################
    #######################################################################################
    ########################           Poisson          ###################################
    #######################################################################################
    #######################################################################################

    # print('Testing Poisson')
    # p_error = test_poisson()
    # print('\tMax error: ', p_error)
    # plt.show(block=True)

    #######################################################################################
    #######################################################################################
    ########################          Divergence        ###################################
    #######################################################################################
    #######################################################################################

    print("Testing Divergence")
    d_error = test_divergence()
    print('\tMax error: ', d_error)
    plt.show(block=True)

    #######################################################################################
    #######################################################################################
    ########################           Gradient         ###################################
    #######################################################################################
    #######################################################################################
    # print("Testing Gradient")
    # for i in tqdm(range(100)):
    #     g_error = test_gradient()
    # print('\tMax error: ', g_error)
    # plt.show(block=True)