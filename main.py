import imgui
from imgui.integrations.pyglet import PygletProgrammablePipelineRenderer

from glumpy import app

# our modules
# from modules import fluid_np as fluid
from modules.trisolver import TriSolver

# np.set_printoptions(threshold=sys.maxsize)

# Use pyglet as backend
app.use("pyglet", major=4, minor=3)

# Constants
WIDTH = 900
HEIGHT = 900
CELLS = 64

# create window with openGL context
window = app.Window(WIDTH, HEIGHT)

# create renderer of imgui on window
imgui.create_context()
imgui_renderer = PygletProgrammablePipelineRenderer(window.native_window) # pass native pyglet window

# main object
# smoke_grid = fluid.Fluid(WIDTH, HEIGHT, CELLS)
vertex      = 'shaders/fluid.vs'
fragment    = 'shaders/fluid.fs'
solver = TriSolver('./assets/mesh8.obj', vertex, fragment, WIDTH, HEIGHT)

@window.event
def on_draw(dt):
    window.clear()

    solver.update_field(dt)

    # draw smoke first
    solver.draw()

    # Imgui Interface
    imgui.new_frame()

    imgui.begin("Controls")
    _, solver.show_grid = imgui.checkbox("Show Grid", solver.show_grid)
    _, solver.show_vectors = imgui.checkbox("Show Vectors", solver.show_vectors)

    changed, solver.smoke_color = imgui.color_edit3("Smoke Color", *solver.smoke_color)

    if changed:
        solver.update_smoke_color()

    imgui.end()

    # render gui on top of everything
    try:
        imgui.render()
        imgui_renderer.render(imgui.get_draw_data())
    except Exception:
        imgui_renderer.shutdown()

# @window.event
# def on_mouse_drag(x, y, dx, dy, buttons):
#     """The mouse was moved with some buttons pressed."""

#     # Case was right mouse button
#     if buttons == 4:
#         radius = 1
#         if x > WIDTH-radius:
#             x = WIDTH-radius
#         if x < radius:
#             x = radius
#         if y > HEIGHT-radius:
#             y = HEIGHT-radius
#         if y < radius:
#             y = radius
        
#         idrow = int(y/smoke_grid.dx) + 1
#         idcol = int(x/smoke_grid.dy) + 1
        
#         for i in range(-radius, radius):
#             idx = idrow + i
#             for j in range(-radius, radius):
#                 idy = idcol + j
#                 smoke_grid.density_field[idx, idy] = 1.0

#     # Case was left mouse button
#     if buttons == 1:
#         radius = 1
#         if x > WIDTH-radius:
#             x = WIDTH-radius
#         if x < radius:
#             x = radius
#         if y > HEIGHT-radius:
#             y = HEIGHT-radius
#         if y < radius:
#             y = radius
        
#         idrow = int(y/smoke_grid.dx) + 1
#         idcol = int(x/smoke_grid.dy) + 1

#         for i in range(-radius, radius):
#             idx = idrow + i
#             for j in range(-radius, radius):
#                 idy = idcol + j
#                 speed = 100
#                 smoke_grid.velocity_field[idx, idy] = [
#                     speed*dx, speed*-dy
#                 ]

@window.event
def on_show():
    # disable resize on show
    window.native_window.set_minimum_size(WIDTH, HEIGHT)
    window.native_window.set_maximum_size(WIDTH, HEIGHT)


if __name__ == "__main__":
    # run app
    app.run()
