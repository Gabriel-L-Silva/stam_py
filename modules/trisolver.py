from math import pi
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

class TriSolver:
    def __init__(self, mesh, k=12, s=5, d=2, only_knn=True, ghost_distance=0.1, source_force=30):
        self.mesh = TriMesh(mesh, k, s, d, only_knn, ghost_distance)

        self.density = np.zeros((self.mesh.n_points))
        self.vectors = np.zeros((self.mesh.n_points,2))

        self.Interpolator = Interpolator(self.mesh)

        self.init_poisson_weights()

        self.source_cells = set()
        self.source_force = source_force

        self.div_history = []

    def apply_boundary_condition(self, field=None):
        if type(field) == type(None):
            self.vectors[list(self.mesh.boundary)] = [0,0]
        else:
            field[list(self.mesh.boundary)] = 0
                    

    def computeExternalForces(self, dt):
        self.apply_boundary_condition()

    def computeViscosity(self, dt):
        self.apply_boundary_condition()

    def computePressure(self, dt):
        x0 = self.poisson_solver()
        grad = self.gradient(x0)
        
        self.vectors -= grad

        self.apply_boundary_condition()


    def computeAdvection(self, dt, new_pos=None):
        if new_pos is None:
            new_pos = self.mesh.points - self.vectors*dt
            new_pos = self.mesh.intersect_boundary(new_pos)
            self.vectors[:,0] = self.Interpolator(self.vectors[:,0], new_pos)
            self.vectors[:,1] = self.Interpolator(self.vectors[:,1], new_pos)
        else:
            self.density = np.clip(self.Interpolator(self.density, new_pos), 0, 1)

        self.apply_boundary_condition()
        return new_pos
    
    def computeSource(self, dt, frame):
        if frame <= 500:
            self.vectors[list(self.source_cells)] = [0, self.source_force ]
            self.density[list(self.source_cells)] = 1
        
        self.apply_boundary_condition()

    def divergence(self, pid):
        div = np.sum(self.mesh.rbf[pid][:,1]*self.vectors[self.mesh.nring[pid],0] 
                   + self.mesh.rbf[pid][:,2]*self.vectors[self.mesh.nring[pid],1])
        return div
        
    def gradient(self, lapl):
        grad = np.zeros((self.mesh.n_points,2))
        for pid in range(self.mesh.n_points):        
            grad[pid] = (np.sum(self.mesh.rbf[pid][:,1]*lapl[self.mesh.nring[pid]]),
                        np.sum(self.mesh.rbf[pid][:,2]*lapl[self.mesh.nring[pid]]))
        return grad

    def init_poisson_weights(self):
        data = []
        data_row = []
        data_col = []
        print('Building Poisson matrix...')
        for pid in tqdm(range(self.mesh.n_points)): 
            if pid in self.mesh.boundary:
                #removing ghost from rbf weights
                w_ghost = self.mesh.rbf[pid][-1]
                self.mesh.rbf[pid] = np.delete(self.mesh.rbf[pid],-1,0)

                w_lap_ghost = w_ghost[0]
                w_n_ghost = w_ghost[1]*self.mesh.normals[pid,0] + w_ghost[2]*self.mesh.normals[pid,1]

                ghost_constant = 1 - w_lap_ghost/w_n_ghost
                weights = (self.mesh.rbf[pid][:,1]*self.mesh.normals[pid,0]
                        + self.mesh.rbf[pid][:,2]*self.mesh.normals[pid,1])
                weights = weights*ghost_constant
            else:
                weights = self.mesh.rbf[pid][:,0]
            
            data[0:0] = weights
            data_row[0:0] = ([pid]*len(self.mesh.nring[pid]))
            data_col[0:0] = self.mesh.nring[pid]
        self.w = csr_matrix((data, (data_row, data_col)), 
                          shape = (self.mesh.n_points, self.mesh.n_points))

        
    def poisson_solver(self, testing=None):        
        if testing != None:
            b = testing
        else:
            b = np.array([self.divergence(pid) if pid not in self.mesh.boundary else 0.0 for pid in range(self.mesh.n_points)])
        lapl = spsolve(self.w, b)
        assert(not np.isnan(lapl).any())
        return lapl

    def velocityStep(self, dt):
        self.computeExternalForces(dt)

        self.computeViscosity(dt)

        self.computePressure(dt)

        new_pos = self.computeAdvection(dt)

        self.computePressure(dt)

        div = np.array([self.divergence(pid) if pid not in self.mesh.boundary else 0.0 for pid in range(self.mesh.n_points)])
        # assert np.allclose(max(abs(div)), 0, atol=1e-2), f'Pressure projection failed, div = {max(abs(div))}'
        self.div_history.append(div)
        return new_pos

    def densityStep(self, dt, frame, new_pos):
        self.computeSource(dt, frame)

        self.computeViscosity(dt)

        self.computeAdvection(dt, new_pos)

    def update_fields(self, dt, frame):
        self.apply_boundary_condition()
        new_pos = self.velocityStep(dt)
        self.apply_boundary_condition()
        self.densityStep(dt, frame, new_pos)