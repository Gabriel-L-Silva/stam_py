from glumpy import glm, gl, gloo
import numpy as np

from modules.trisolver import TriSolver

class SimulationWindow:
    def __init__(self, solver: TriSolver, f_vertex = None, f_fragment = None, q_vertex = None, q_fragment = None, q_geometry = None) -> None:
        
        self.paused = True
        self.frame = 0
        self.save_video = True
        self.speed = 0.1

        self.solver = solver
        
        self.show_vectors = True
        self.show_grid = True
        self.smoke_color = [1,1,1]
        self.view_matrix = [0,0,-5]
        self.grid_color = [1,0,0]
        self.quiv_color = [0,1,0,1]

        self.program = gloo.Program(f_vertex, f_fragment, version='430')
        self.idx_buff = self.solver.mesh.faces.flatten().astype(np.uint32).view(gloo.IndexBuffer)
        self.program['position'] = self.solver.mesh.points + [-1,-1]
        self.program['color'] = self.smoke_color

        self.quiver_program = gloo.Program(q_vertex, q_fragment, q_geometry, version='430', count=self.solver.mesh.n_points)
        self.quiver_program['position'] = self.solver.mesh.points + [-1,-1]
        self.quiver_program['acolor'] = self.quiv_color    
        self.quiver_program['Xvelocity'] = self.solver.vectors[:,0]
        self.quiver_program['Yvelocity'] = self.solver.vectors[:,1]
        
    def draw(self):
        self.program['density'] = self.solver.density
        self.program.draw(gl.GL_TRIANGLES, self.idx_buff)

        if self.show_vectors:
            self.draw_vectors()
        if self.show_grid:
            self.draw_grid()


    def draw_grid(self):
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        self.program['color'] = self.grid_color
        self.program['density'] = 1.0
        self.program.draw(gl.GL_TRIANGLES, self.idx_buff)
        self.program['color'] = self.smoke_color
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

    def draw_vectors(self):
        self.quiver_program['Xvelocity'] = self.solver.vectors[:,0]
        self.quiver_program['Yvelocity'] = self.solver.vectors[:,1]
        self.quiver_program.draw(gl.GL_POINTS)

    def update_smoke_color(self):
        self.program["color"] = self.smoke_color

    def update_view_matrix(self):
        self.program["u_view"] = glm.translate(np.eye(4), self.view_matrix[0], self.view_matrix[1], self.view_matrix[2])
        self.quiver_program["u_view"] = glm.translate(np.eye(4), self.view_matrix[0], self.view_matrix[1], self.view_matrix[2])

    def advance_frame(self, dt):
        self.frame += 1
        if self.frame == 1000:
            self.paused = True
        self.solver.update_fields(dt, self.frame)
    
    def build_scene_1(self, dt):
        dx = 1/64.0
        xx = np.arange(0.5*np.pi-dx, 0.5*np.pi+dx, dx/2)
        yy = np.arange(0.15*np.pi, 0.2*np.pi, dx/2)
        cells = np.unique([int(self.solver.mesh.triFinder(x,y)) for x,y in np.array(np.meshgrid(xx, yy)).T.reshape(-1, 2)])
        for p in self.solver.mesh.faces[cells]:
            for c in p:
                self.solver.source_cells.add(c)
        self.solver.computeSource(dt, self.frame)