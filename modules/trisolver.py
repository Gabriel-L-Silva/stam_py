from math import pi
import numpy as np
from modules.trimesh import TriMesh
from modules.interpolator import Interpolator
from glumpy import app, gloo, gl

class TriSolver:
    def __init__(self, filename, vertex, fragment, WIDTH, HEIGHT):
        self.mesh = TriMesh(filename)

        self.show_vectors = False
        self.show_grid = False
        self.smoke_color = [1,1,1]
        self.grid_color = [1,0,0]
        self.quiv_color = [0,1,0]

        self.program = gloo.Program(vertex, fragment)
        self.idx_buff = self.mesh.faces.flatten().astype(np.uint32).view(gloo.IndexBuffer)
        self.program['position'] = self.mesh.points + [-1,-1]
        self.program['color'] = self.smoke_color

        self.quiver_program = gloo.Program('shaders/quiver.vs', 'shaders/quiver.fs')
        self.quiver_program['position'] = self.mesh.points + [-1,-1]
        self.quiver_program["iResolution"] = WIDTH, HEIGHT
        self.quiver_program["dim_x"] = np.sqrt(self.mesh.n_points).astype(np.int)
        self.quiver_program["dim_y"] = np.sqrt(self.mesh.n_points).astype(np.int)
        self.quiver_program["linewidth"] = 1.0

        self.density = np.random.random((self.mesh.n_points))
        self.vectors = np.random.random((self.mesh.n_points,2))
        
        self.Xinterpolator = Interpolator(self.mesh, self.vectors[:,0])
        self.Yinterpolator = Interpolator(self.mesh, self.vectors[:,1])
        self.Densinterpolator = Interpolator(self.mesh, self.density)
        # px = [mesh.points[b][0] for b in boundary]
        # py = [mesh.points[b][1] for b in boundary]
        # plt.scatter(px,py)
    
    def draw(self, *args):
        self.program['density'] = self.density
        self.program.draw(gl.GL_TRIANGLES, self.idx_buff, *args)

        if self.show_grid:
            self.draw_grid()

        if self.show_vectors:
            self.draw_vectors()

    def draw_grid(self):
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        self.program['color'] = self.grid_color
        self.program.draw(gl.GL_TRIANGLES, self.idx_buff)
        self.program['color'] = self.smoke_color
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

    def draw_vectors(self):
        self.quiver_program["velocities"] = self.vectors.view(gloo.TextureFloat2D)
        self.quiver_program.draw(gl.GL_TRIANGLE_STRIP)

    def update_smoke_color(self):
        self.program["color"] = self.smoke_color

    def apply_boundary_condition(self):
        self.vectors[self.mesh.left_id,0] = 0
        self.vectors[self.mesh.right_id,0] = 0
        self.vectors[self.mesh.top_id,1] = 0
        self.vectors[self.mesh.bottom_id,1] = 0

    def computeExternalForces(self, dt):
        pass

    def computeViscosity(self, dt):
        pass

    def computePressure(self, dt):
        x0 = self.poisson_solver()
        grad = self.gradient(x0)

        self.vectors[:,0] -= grad[:,0]
        self.vectors[:,1] -= grad[:,1]

        self.apply_boundary_condition()

    def computeAdvection(self, density, dt):
        new_pos = self.mesh.points - self.vectors*dt
        new_pos = np.clip(new_pos,0,pi)
        if density:
            self.density = self.Densinterpolator(new_pos)
        else:
            self.vectors[:,0] = self.Xinterpolator(new_pos)
            self.vectors[:,1] = self.Yinterpolator(new_pos)

        self.apply_boundary_condition()

    def computeSource(self, dt):
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

    def velocityStep(self, dt):
        self.computeExternalForces(dt)

        self.computeViscosity(dt)

        self.computePressure(dt)

        self.computeAdvection(False, dt)

        # computePressure()

    def densityStep(self, dt):
        self.computeSource(dt)

        self.computeViscosity(dt)

        self.computeAdvection(True, dt)

    def update_field(self, dt):
        self.apply_boundary_condition()
        self.velocityStep(dt)

        self.apply_boundary_condition()
        self.densityStep(dt)