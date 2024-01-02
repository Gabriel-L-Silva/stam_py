import math
from numpy import pi
import imgui
from imgui.integrations.pyglet import PygletProgrammablePipelineRenderer
from tqdm import tqdm
import pyglet

from glumpy import app, glm, gloo
# our modules
from modules.simulationWindow import SimulationWindow
from modules.tri_mesh import TriMesh

import numpy as np

# Use pyglet as backend
app.use("pyglet", major=4, minor=3)
config = app.configuration.Configuration()

config.major_version = 4
config.minor_version = 3

# Constants
WIDTH = 720
HEIGHT = 720

# create window with openGL context
window = app.Window(WIDTH, HEIGHT,config=config)

# create renderer of imgui on window
imgui.create_context()
imgui_renderer = PygletProgrammablePipelineRenderer(window.native_window) # pass native pyglet window

# main objects
f_vertex      = 'shaders/fluid.vs'
f_fragment    = 'shaders/fluid.fs'
q_vertex      = 'shaders/quiver.vs'
q_fragment    = 'shaders/quiver.fs'
q_geometry    = 'shaders/quiver.gs'
view = np.eye(4)
model = np.eye(4)
projection = glm.perspective(45.0, 1, 0.1, 10000.0)
h = math.tan(45.0 / 360.0 * math.pi) * 0.1
glm.translate(view, 0,0,-3)
view_matrix = [0,0,-3]
simWindow = SimulationWindow(5, view, model, projection, view_matrix, f_vertex, f_fragment, q_vertex, q_fragment, q_geometry)
frames = []

def preview_mesh():
    simWindow.draw_grid()

@window.event
def on_draw(dt):
    window.clear()

    simWindow.draw_gui()
    if simWindow.ready:
        # draw smoke first
        simWindow.draw()
        if not simWindow.paused or simWindow.next_frame:
            if simWindow.save_video:
                dt = 1/60.0
                frames.append(np.asarray(pyglet.image.get_buffer_manager().get_color_buffer().get_image_data().get_data()))
            profiler.enable()
            simWindow.advance_frame(dt)
            profiler.disable()
    else:
        preview_mesh()

    # render gui on top of everything
    try:
        imgui.render()
        imgui_renderer.render(imgui.get_draw_data())
    except Exception:
        imgui_renderer.shutdown()


def mouse_coord_to_world(x, y):
    IP = np.linalg.inv(simWindow.projection)
    IV = np.linalg.inv(simWindow.view)
    IVP = IV * IP

    norm_x = 2*x/WIDTH - 1
    norm_y = 2*y/HEIGHT - 1

    input = np.array([norm_x, norm_y, -1, 1])

    pos = np.dot(IVP, input)
    return pos[:-1] * pos[-1]

@window.event
def on_mouse_press(x, y, button):
    if not simWindow.ready:
        return 
    
    world_coord = mouse_coord_to_world(x, y)[:-1]
    # Case was right mouse button
    if button == 4:        
        cell = simWindow.solver.mesh.triFinder(*world_coord)
        
        simWindow.solver.density[simWindow.solver.mesh.faces[cell]] = 1.0
    
    if button == 2:
        cell = simWindow.solver.mesh.triFinder(*world_coord)
        simWindow.solver.density[simWindow.solver.mesh.faces[cell]] = 1
        simWindow.solver.vectors[simWindow.solver.mesh.faces[cell],:2] = [0,simWindow.source_force]
        for c in simWindow.solver.mesh.faces[cell]:
            simWindow.solver.source_cells.add(c)

@window.event
def on_mouse_drag(x, y, dx, dy, buttons):
    """The mouse was moved with some buttons pressed."""
    if not simWindow.ready:
        return 
    
    world_coord = mouse_coord_to_world(x, y)[:-1]
    # Case was right mouse button
    if buttons == 4:

        cell = simWindow.solver.mesh.triFinder(*world_coord)
        
        simWindow.solver.density[simWindow.solver.mesh.faces[cell]] = 1.0

    # Case was left mouse button
    if buttons == 1:       

        cell = simWindow.solver.mesh.triFinder(*world_coord)
        
        simWindow.solver.vectors[simWindow.solver.mesh.faces[cell],:2] += [
            simWindow.speed*dx, simWindow.speed*-dy
        ]

    if buttons == 2:
        cell = simWindow.solver.mesh.triFinder(*world_coord)
        simWindow.solver.density[simWindow.solver.mesh.faces[cell]] = 1
        simWindow.solver.vectors[simWindow.solver.mesh.faces[cell],:2] = [0,simWindow.source_force]
        for c in simWindow.solver.mesh.faces[cell]:
            simWindow.solver.source_cells.add(c)
@window.event    
def on_mouse_scroll(x, y, dx, dy):
    'The mouse wheel was scrolled by (dx,dy).'

    simWindow.view_matrix[-1] -= dy
    simWindow.update_view_matrix()

@window.event
def on_show():
    # disable resize on show
    window.native_window.set_minimum_size(WIDTH, HEIGHT)
    window.native_window.set_maximum_size(WIDTH, HEIGHT)


if __name__ == "__main__":
    import cv2
    import os
    from PIL import Image
    # run app
    import cProfile, pstats
    global profiler
    profiler = cProfile.Profile()

    try:
        app.run()
    except AttributeError:
        if simWindow.save_video and len(frames) > 0:
            print('saving frames')
            video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 60, (WIDTH,HEIGHT))
            for f, frame  in tqdm(enumerate(frames)):
                if len(frame) != WIDTH*HEIGHT*4:
                    break
                video.write(cv2.cvtColor(np.array(Image.frombuffer("RGBA", (WIDTH, HEIGHT), frame, "raw", "RGBA", 0, -1)), cv2.COLOR_RGBA2BGR))
            video.release()
        print('acabou')
        stats = pstats.Stats(profiler).sort_stats('tottime')
        stats.print_stats()