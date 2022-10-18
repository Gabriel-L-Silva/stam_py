from glumpy import glm, gl, gloo
import numpy as np
import imgui
from modules.trisolver import TriSolver
from modules.generate_mesh import get_geojson, polygon_triangulation
from shapely.geometry import Polygon, shape

class SimulationWindow:
    def __init__(self,  view, model, projection, view_matrix, f_vertex = None, f_fragment = None, q_vertex = None, q_fragment = None, q_geometry = None) -> None:

        self.paused = True
        self.next_frame = True
        self.frame = 0
        self.save_video = True
        self.speed = 0.1

        self.ready = False
        
        self.show_vectors = True
        self.show_grid = True
        self.smoke_color = [1,1,1]
        self.view_matrix = [0,0,-5]
        self.grid_color = [1,0,0]
        self.quiv_color = [0,1,0,1]

        self.view = view
        self.model = model
        self.projection = projection
        self.view_matrix = view_matrix

        self.f_vertex = f_vertex
        self.f_fragment = f_fragment

        self.program = gloo.Program(f_vertex, f_fragment, version='430')
        self.program['color'] = self.smoke_color
        self.program['u_model'] = model
        self.program['u_view'] = view
        self.program['u_projection'] = projection

        self.quiver_program = gloo.Program(q_vertex, q_fragment, q_geometry, version='430')
        self.quiver_program['acolor'] = self.quiv_color    
        self.quiver_program['u_model'] = model
        self.quiver_program['u_view'] = view
        self.quiver_program['u_projection'] = projection

        self.current_mesh: int = 0
        self.geojson, self.mesh_list = get_geojson()
        self.mesh_list = list(self.mesh_list)
        
        self.update_mesh()
        
    def build_solver(self):
        self.solver: TriSolver = TriSolver(self.meshpath)
        self.program['position'] = self.solver.mesh.points + [-1,-1]
        self.idx_buff = self.solver.mesh.faces.flatten().astype(np.uint32).view(gloo.IndexBuffer)
        self.quiver_program['position'] = self.solver.mesh.points + [-1,-1]
        self.quiver_program['Xvelocity'] = self.solver.vectors[:,0]
        self.quiver_program['Yvelocity'] = self.solver.vectors[:,1]

        self.ready = True

    def draw(self):
        self.program['density'] = self.solver.density
        self.program.draw(gl.GL_TRIANGLES, self.idx_buff)

        if self.show_vectors:
            self.draw_vectors()
        if self.show_grid:
            self.draw_grid()

    def update_mesh(self):
        poly: Polygon = shape(self.geojson[self.current_mesh]['geometry'])
        self.mesh = polygon_triangulation(poly)
        # self.mesh.show(smooth=False, flags={'wireframe':True})
        self.program = gloo.Program(self.f_vertex, self.f_fragment, version='430')
        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.program['u_projection'] = self.projection
        self.program['color'] = self.smoke_color
        self.idx_buff = self.mesh.faces.flatten().astype(np.uint32).view(gloo.IndexBuffer)
        normalized_x = 2* (self.mesh.vertices[:,0] - np.min(self.mesh.vertices[:,0]))/np.ptp(self.mesh.vertices[:,0]) - 1
        normalized_y = 2* (self.mesh.vertices[:,1] - np.min(self.mesh.vertices[:,1]))/np.ptp(self.mesh.vertices[:,1]) - 1
        self.program['position'] = np.stack((normalized_x, normalized_y),axis=1)

    def draw_gui(self):
        # Imgui Interface
        imgui.new_frame()
        imgui.begin("Controls")
        imgui.text(f"Frame count: {self.frame}")
        if self.ready:
            clicked = imgui.button("Pause")
            if clicked:
                self.paused = not self.paused
            self.advance_frame = imgui.button("Advance frame")
            _, self.show_grid = imgui.checkbox("Show Grid", self.show_grid)
            _, self.show_vectors = imgui.checkbox("Show Vectors", self.show_vectors)
            _, self.save_video = imgui.checkbox("Save video", self.save_video)

            changed, self.smoke_color = imgui.color_edit3("Smoke Color", *self.smoke_color)
            if changed:
                self.update_smoke_color()
            

            changed, sp = imgui.drag_float("Speed", self.speed, change_speed=0.1)
            if changed:
                self.speed = sp
        else:
            if imgui.collapsing_header('Solver'):
                pass
            clicked, self.current_mesh = imgui.combo('Mesh selector', self.current_mesh, self.mesh_list)
            if clicked:
                self.update_mesh()
            changed,  vm = imgui.drag_float3("View Matrix", *self.view_matrix, change_speed=0.01)
            self.view_matrix = list(vm)
            if changed:
                self.update_view_matrix()
        imgui.end()

    def draw_grid(self):
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        self.program['color'] = self.grid_color
        self.program['density'] = np.repeat(1.0, len(self.mesh.vertices))
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
        self.solver.update_fields(dt, self.frame)