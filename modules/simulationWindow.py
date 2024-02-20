from glumpy import glm, gl, gloo
import numpy as np
import imgui
import trimesh
from modules.tri_mesh import TriMesh
from modules.trisolver import TriSolver
from modules.generate_mesh import get_geojson, polygon_triangulation
from shapely.geometry import Polygon, shape, MultiPolygon, box
import os

class SimulationWindow:
    def __init__(self, source_force, view, model, projection, view_matrix, f_vertex = None, f_fragment = None, q_vertex = None, q_fragment = None, q_geometry = None) -> None:
        self.source_force = source_force

        self.paused = True
        self.next_frame = False
        self.frame = 0
        self.save_video = True
        self.speed = 0.1

        self.adaptative_timestep = False
        self.n_timesteps = 1

        self.ready = False
        
        self.triangulate = False 
        self.resolution = 1

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

        self.obj_assets_names = [f for f in os.listdir('./assets') if f.endswith('.obj')]
        self.mesh_list = [x for x in self.obj_assets_names]

        # self.geojson, self.geo_list = get_geojson()
        # for x in self.geo_list:
        #     self.mesh_list.append(x)
        
        self.current_mesh: int = 0
        self.update_mesh()
        
    def build_solver(self):
        self.solver = TriSolver(self.mesh)
        self.quiver_program['position'] = self.solver.mesh.points
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
        if self.current_mesh >= len(self.obj_assets_names):
            self.mesh: Polygon = shape(self.geojson[self.current_mesh-len(self.obj_assets_names)]['geometry'])
            if type(poly) == MultiPolygon:
                poly = list(poly)[0]
        elif 'donut' in self.obj_assets_names[self.current_mesh]:
            if '0-04' in self.obj_assets_names[self.current_mesh]:
                self.mesh =  trimesh.load_mesh(f'./assets/{self.obj_assets_names[self.current_mesh]}')
                self.mesh.holes = [np.arange(128)]
                self.mesh.exterior = np.arange(128,192)
                poly = Polygon(self.mesh.vertices[:128], [self.mesh.vertices[128:192]])
            elif '00401' in self.obj_assets_names[self.current_mesh]:
                self.mesh =  trimesh.load_mesh(f'./assets/{self.obj_assets_names[self.current_mesh]}')
                self.mesh.holes = [np.arange(128,192)]
                self.mesh.exterior = np.arange(128)
                poly = Polygon(self.mesh.vertices[self.mesh.exterior], [self.mesh.vertices[hole] for hole in self.mesh.holes])
            elif '5' in self.obj_assets_names[self.current_mesh]:
                self.mesh =  trimesh.load_mesh(f'./assets/{self.obj_assets_names[self.current_mesh]}')
                self.mesh.holes = [np.arange(300,350)]
                self.mesh.exterior = np.arange(300)
                poly = Polygon(self.mesh.vertices[self.mesh.exterior], [self.mesh.vertices[hole] for hole in self.mesh.holes])
            elif '3(100)' in self.obj_assets_names[self.current_mesh]:
                self.mesh =  trimesh.load_mesh(f'./assets/{self.obj_assets_names[self.current_mesh]}')
                self.mesh.holes = [np.arange(300,400)]
                self.mesh.exterior = np.arange(300)
                poly = Polygon(self.mesh.vertices[self.mesh.exterior], [self.mesh.vertices[hole] for hole in self.mesh.holes])
            elif '2' in self.obj_assets_names[self.current_mesh]:
                self.mesh =  trimesh.load_mesh(f'./assets/{self.obj_assets_names[self.current_mesh]}')
                self.mesh.holes = [np.arange(128,191)]
                self.mesh.exterior = list(range(128))+list(range(192,320))
                poly = Polygon(self.mesh.vertices[:128], [self.mesh.vertices[128:192]])
            elif '1' in self.obj_assets_names[self.current_mesh]:
                pass
            elif 'hfun' in self.obj_assets_names[self.current_mesh]:
                self.mesh =  trimesh.load_mesh(f'./assets/{self.obj_assets_names[self.current_mesh]}')
                self.mesh.holes = [np.arange(300,400)]
                self.mesh.exterior = np.arange(300)
                poly = Polygon(self.mesh.vertices[self.mesh.exterior], [self.mesh.vertices[hole] for hole in self.mesh.holes])

            filename = 'donut'
        else:
            self.mesh = trimesh.load_mesh(f'./assets/{self.obj_assets_names[self.current_mesh]}')
            poly = Polygon(self.mesh.vertices[:,:2][:256])
            self.mesh.holes = []
            self.mesh.exterior = np.arange(256)
            filename = 'circle'
        if self.triangulate:
            self.mesh = polygon_triangulation(poly, self.resolution)
            self.mesh.metadata['file_name'] = filename
            self.mesh.boundary = poly
        
        self.program = gloo.Program(self.f_vertex, self.f_fragment, version='430')
        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.program['u_projection'] = self.projection
        self.program['color'] = self.smoke_color
        self.idx_buff = self.mesh.faces.flatten().astype(np.uint32).view(gloo.IndexBuffer)
        self.program['position'] = self.mesh.vertices[:,:2]
        return self.mesh

    def draw_gui(self):
        # Imgui Interface
        imgui.new_frame()
        imgui.begin("Controls")
        imgui.text(f"Frame count: {self.frame}")
        if self.ready:
            clicked = imgui.button("Pause")
            if clicked:
                self.paused = not self.paused
            self.next_frame = imgui.button("Advance frame")
            _, self.show_grid = imgui.checkbox("Show Grid", self.show_grid)
            _, self.show_vectors = imgui.checkbox("Show Vectors", self.show_vectors)

            changed, self.smoke_color = imgui.color_edit3("Smoke Color", *self.smoke_color)
            if changed:
                self.update_smoke_color()
            

            changed, sp = imgui.drag_float("Speed", self.speed, change_speed=0.1)
            if changed:
                self.speed = sp
        else:
            if imgui.collapsing_header('Solver'):
                clicked, self.current_mesh = imgui.combo('Mesh selector', self.current_mesh, self.mesh_list)
                if clicked:
                    self.update_mesh()
            changed,  vm = imgui.drag_float3("View Matrix", *self.view_matrix, change_speed=1.0)
            self.view_matrix = list(vm)
            if changed:
                self.update_view_matrix()
            
            _, self.triangulate = imgui.checkbox("Force Triangulation", self.triangulate)
            if self.triangulate:
                changed, self.resolution = imgui.drag_float("Resolution", self.resolution, change_speed=0.1)
            self.update_mesh()

            _, self.adaptative_timestep = imgui.checkbox("Adaptative timestep", self.adaptative_timestep)
            if self.adaptative_timestep:
                changed, self.n_timesteps = imgui.drag_int("Num of substeps", self.n_timesteps, change_speed=0.1)

            _, self.save_video = imgui.checkbox("Save video", self.save_video)
            if imgui.button("Build Solver"):
                self.build_solver()
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
        self.view = glm.translate(np.eye(4), self.view_matrix[0], self.view_matrix[1], self.view_matrix[2])
        self.program["u_view"] = glm.translate(np.eye(4), self.view_matrix[0], self.view_matrix[1], self.view_matrix[2])
        self.quiver_program["u_view"] = glm.translate(np.eye(4), self.view_matrix[0], self.view_matrix[1], self.view_matrix[2])

    def advance_frame(self, dt):
        self.frame += 1
        sim_dt = dt / self.n_timesteps
        for _ in range(self.n_timesteps):
            self.solver.update_fields(sim_dt, self.frame)

        if self.frame == 1000:
            self.paused = True