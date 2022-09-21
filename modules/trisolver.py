from math import pi
import numpy as np
from modules.trimesh import TriMesh
from modules.interpolator import Interpolator
class TriSolver:
    def __init__(self, filename, timeInterval):
        self.mesh = TriMesh(filename)

        self.timeInterval = timeInterval
        self.density = np.zeros((self.mesh.n_points))
        self.vectors = np.random.random((self.mesh.n_points,2))
        
        self.Xinterpolator = Interpolator(self.mesh, self.vectors[:,0])
        self.Yinterpolator = Interpolator(self.mesh, self.vectors[:,1])
        self.Densinterpolator = Interpolator(self.mesh, self.density)
        # px = [mesh.points[b][0] for b in boundary]
        # py = [mesh.points[b][1] for b in boundary]
        # plt.scatter(px,py)

    def apply_boundary_condition(self):
        self.vectors[self.mesh.left_id,0] = 0
        self.vectors[self.mesh.right_id,0] = 0
        self.vectors[self.mesh.top_id,1] = 0
        self.vectors[self.mesh.bottom_id,1] = 0

    def computeExternalForces(self):
        pass

    def computeViscosity(self):
        pass

    def computePressure(self):
        x0 = self.poisson_solver()
        grad = self.gradient(x0)

        self.vectors[:,0] -= grad[:,0]
        self.vectors[:,1] -= grad[:,1]

        self.apply_boundary_condition()

    def computeAdvection(self, density):
        new_pos = self.mesh.points - self.vectors*self.timeInterval
        new_pos = np.clip(new_pos,0,pi)
        if density:
            self.density = self.Densinterpolator(new_pos)
        else:
            self.vectors[:,0] = self.Xinterpolator(new_pos)
            self.vectors[:,1] = self.Yinterpolator(new_pos)

        self.apply_boundary_condition()

    def computeSource(self):
        self.density[[25,40]] = 1

    def divergent(self, pid):
        div = np.sum(self.mesh.rbf[pid][:,1]*self.vectors[self.mesh.nring[pid],0] 
                   + self.mesh.rbf[pid][:,2]*self.vectors[self.mesh.nring[pid],1])
        self.apply_boundary_condition()
        return div
        

    def gradient(self, lapl):
        grad = np.zeros((self.mesh.n_points,2))
        for pid in range(self.mesh.n_points):        
            grad[pid] = (np.sum(self.mesh.rbf[pid][:,1]*lapl[self.mesh.nring[pid]]),
                        np.sum(self.mesh.rbf[pid][:,2]*lapl[self.mesh.nring[pid]]))
        return grad

    def poisson_solver(self):
        lapl = np.asarray([])
        w = np.zeros(self.mesh.n_points)
        b = np.zeros(self.mesh.n_points)
        for pid in range(self.mesh.n_points): 
            if pid in self.mesh.boundary:
                weights = (self.mesh.rbf[pid][:,1]*self.mesh.normals[pid,0] 
                        + self.mesh.rbf[pid][:,2]*self.mesh.normals[pid,1])
            else:
                weights = self.mesh.rbf[pid][:,0]
                b[pid] = self.divergent(pid)
            
            line = np.zeros(self.mesh.n_points)
            line[self.mesh.nring[pid]]= weights
            # p.show(cpos='xy')
            w = np.vstack([w, line])
        w = np.delete(w, 0,0)
        # w[0] = np.identity(mesh.n_points)[0]
        # b[0] = solution(mesh.points[0][0],mesh.points[0][1])
        # np.savetxt("foo.csv", w, fmt="%.7s",delimiter="\t")

        lapl = np.linalg.solve(w,b)
        return lapl

    def velocityStep(self):
        self.computeExternalForces()

        self.computeViscosity()

        self.computePressure()

        self.computeAdvection(False)

        # computePressure()

    def densityStep(self):
        self.computeSource()

        self.computeViscosity()

        self.computeAdvection(True)

    def update_field(self):
        self.apply_boundary_condition()
        self.velocityStep()

        self.apply_boundary_condition()
        self.densityStep()