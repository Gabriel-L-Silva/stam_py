from numpy import pi
import imgui
from imgui.integrations.pyglet import PygletProgrammablePipelineRenderer

from glumpy import app, glm, gloo, gl
from glumpy.transforms import PVMProjection
# our modules
# from modules import fluid_np as fluid
# from trisolver import TriSolver

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
vertex      = 'shaders/quiver.vs'
fragment    = 'shaders/quiver.fs'
geometry    = 'shaders/quiver.gs'

program = gloo.Program(vertex,fragment,geometry,version='330')

points = [[-0.5, 0.5],[0.5, 0.5],[0.5, -0.5],[-0.5, -0.5]]
velocity =np.asarray([[-0, 0],[0, 0],[0, -0],[-0, -0]])
color = [1,0,0,1]
@window.event
def on_init():
    view = np.eye(4,dtype=np.float32)
    model = np.eye(4,dtype=np.float32)
    projection = glm.perspective(45.0, 1, 2.0, 100.0)
    glm.translate(view, -0.57,-0.57,-3.8)
    # program['mvp'] = projection*view*model
    program['position'] = points
    program['Xvelocity'] = velocity[:,0]
    program['Yvelocity'] = velocity[:,1]
    program['vec_length'] = 0.1
    program['acolor'] = color
    program['u_model'] = model
    program['u_view'] = view
    program['u_projection'] = glm.perspective(45.0, 1, 2.0, 100.0)

@window.event
def on_draw(dt):
    window.clear()
    # Imgui Interface
    imgui.new_frame()

    imgui.begin("Controls")
    program.draw(gl.GL_POINTS)
    imgui.end()

    # render gui on top of everything
    try:
        imgui.render()
        imgui_renderer.render(imgui.get_draw_data())
    except Exception:
        imgui_renderer.shutdown()


@window.event
def on_show():
    # disable resize on show
    window.native_window.set_minimum_size(WIDTH, HEIGHT)
    window.native_window.set_maximum_size(WIDTH, HEIGHT)


if __name__ == "__main__":
    app.run()
