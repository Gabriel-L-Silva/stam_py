from numpy import pi
import imgui
from imgui.integrations.pyglet import PygletProgrammablePipelineRenderer

from glumpy import app, glm, gl
from glumpy.transforms import PVMProjection
from modules.simulationWindow import SimulationWindow
# our modules
# from modules import fluid_np as fluid
from modules.trisolver import TriSolver

import numpy as np
# np.set_printoptions(threshold=sys.maxsize)

# Use pyglet as backend
app.use("pyglet", major=4, minor=3)
config = app.configuration.Configuration()

config.major_version = 4
config.minor_version = 3

# Constants
WIDTH = 900
HEIGHT = 900
CELLS = 64

# create window with openGL context
window = app.Window(WIDTH, HEIGHT,config=config)

# create renderer of imgui on window
imgui.create_context()
imgui_renderer = PygletProgrammablePipelineRenderer(window.native_window) # pass native pyglet window

# main object
# smoke_grid = fluid.Fluid(WIDTH, HEIGHT, CELLS)
f_vertex      = 'shaders/fluid.vs'
f_fragment    = 'shaders/fluid.fs'
q_vertex      = 'shaders/quiver.vs'
q_fragment    = 'shaders/quiver.fs'
q_geometry    = 'shaders/quiver.gs'

solver = TriSolver('./assets/mesh8.obj')
simWindow = SimulationWindow(solver, f_vertex, f_fragment, q_vertex, q_fragment, q_geometry)

@window.event
def on_init():

    view = np.eye(4,dtype=np.float32)
    model = np.eye(4,dtype=np.float32)
    projection = glm.perspective(45.0, 1, 2.0, 100.0)
    glm.translate(view, -0.57,-0.57,-3.8)
    simWindow.view_matrix = [-0.57,-0.57,-3.8]
    simWindow.program['u_model'] = model
    simWindow.program['u_view'] = view
    simWindow.program['u_projection'] = projection
    simWindow.quiver_program['u_model'] = model
    simWindow.quiver_program['u_view'] = view
    simWindow.quiver_program['u_projection'] = projection
    # gl.glEnable(gl.GL_DEPTH_TEST)

@window.event
def on_draw(dt):
    window.clear()
    
    
    # draw smoke first
    simWindow.draw()
    advance_frame = True

    # Imgui Interface
    imgui.new_frame()

    imgui.begin("Controls")
    clicked = imgui.button("Pause")
    if clicked:
        simWindow.paused = not simWindow.paused
    advance_frame = imgui.button("Advance frame")
    _, simWindow.show_grid = imgui.checkbox("Show Grid", simWindow.show_grid)
    _, simWindow.show_vectors = imgui.checkbox("Show Vectors", simWindow.show_vectors)

    changed, simWindow.smoke_color = imgui.color_edit3("Smoke Color", *simWindow.smoke_color)

    if changed:
        simWindow.update_smoke_color()
    
    changed,  vm= imgui.drag_float3("View Matrix", *simWindow.view_matrix, change_speed=0.01)
    simWindow.view_matrix = list(vm)
    if changed:
        simWindow.update_view_matrix()

    imgui.end()

    if not simWindow.paused or advance_frame:
        simWindow.advance_frame(dt)
        print(solver.vectors.max())
    # render gui on top of everything
    try:
        imgui.render()
        imgui_renderer.render(imgui.get_draw_data())
    except Exception:
        imgui_renderer.shutdown()

@window.event
def on_mouse_press(x, y, button):
    # Case was right mouse button
    if button == 4:        
        cell = solver.mesh.triFinder(x/WIDTH*pi,y/HEIGHT*pi)
        
        solver.density[solver.mesh.faces[cell]] = 1.0

@window.event
def on_mouse_drag(x, y, dx, dy, buttons):
    """The mouse was moved with some buttons pressed."""

    # Case was right mouse button
    if buttons == 4:

        cell = solver.mesh.triFinder(x/WIDTH*pi,y/HEIGHT*pi)
        
        solver.density[solver.mesh.faces[cell]] = 1.0

    # Case was left mouse button
    if buttons == 1:       

        cell = solver.mesh.triFinder(x/WIDTH*pi,y/HEIGHT*pi)
        speed = 0.01
        solver.vectors[solver.mesh.faces[cell]] += [
            speed*dx, speed*-dy
        ]
@window.event    
def on_mouse_scroll(x, y, dx, dy):
    'The mouse wheel was scrolled by (dx,dy).'
    simWindow.view_matrix[-1] -= dy*0.1   
    simWindow.update_view_matrix()

@window.event
def on_show():
    # disable resize on show
    window.native_window.set_minimum_size(WIDTH, HEIGHT)
    window.native_window.set_maximum_size(WIDTH, HEIGHT)


if __name__ == "__main__":
    # run app
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    # try:
    app.run()
    # except:
    #     print('acabou')
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()
