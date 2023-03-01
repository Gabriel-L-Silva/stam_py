from matplotlib import animation, cm
from numpy import pi
import imgui
from imgui.integrations.pyglet import PygletProgrammablePipelineRenderer
from tqdm import tqdm
import pyglet

from glumpy import app, glm
# our modules
from modules.simulationWindow import SimulationWindow
from modules.trisolver import TriSolver

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
solver = TriSolver('./assets/regular_tri_grid64.obj')
simWindow = SimulationWindow(solver, f_vertex, f_fragment, q_vertex, q_fragment, q_geometry)
frames = []

@window.event
def on_init():
    view = np.eye(4)
    model = np.eye(4)
    projection = glm.perspective(45.0, 1, 2.0, 100.0)
    glm.translate(view, -0.57,-0.57,-3.8)
    simWindow.view_matrix = [-0.57,-0.57,-3.8]
    simWindow.program['u_model'] = model
    simWindow.program['u_view'] = view
    simWindow.program['u_projection'] = projection
    simWindow.quiver_program['u_model'] = model
    simWindow.quiver_program['u_view'] = view
    simWindow.quiver_program['u_projection'] = projection

@window.event
def on_draw(dt):
    window.clear()
    
    # draw smoke first
    simWindow.draw()
    advance_frame = True

    # Imgui Interface
    imgui.new_frame()

    imgui.begin("Controls")
    imgui.text(f"Frame count: {simWindow.frame}")
    clicked = imgui.button("Pause")
    if clicked:
        simWindow.paused = not simWindow.paused
    advance_frame = imgui.button("Advance frame")
    clicked = imgui.button("Build scene 1")
    if clicked:
        simWindow.build_scene_1(dt)
    
    _, simWindow.show_grid = imgui.checkbox("Show Grid", simWindow.show_grid)
    _, simWindow.show_vectors = imgui.checkbox("Show Vectors", simWindow.show_vectors)
    _, simWindow.save_video = imgui.checkbox("Save video", simWindow.save_video)

    changed, simWindow.smoke_color = imgui.color_edit3("Smoke Color", *simWindow.smoke_color)
    if changed:
        simWindow.update_smoke_color()
    
    changed,  vm = imgui.drag_float3("View Matrix", *simWindow.view_matrix, change_speed=0.01)
    simWindow.view_matrix = list(vm)
    if changed:
        simWindow.update_view_matrix()

    changed, sp = imgui.drag_float("Speed", simWindow.speed, change_speed=0.1)
    if changed:
        simWindow.speed = sp
    imgui.end()

    global m
    if not simWindow.paused or advance_frame:
        if simWindow.save_video:
            dt = 1/60.0
            frames.append(np.asarray(pyglet.image.get_buffer_manager().get_color_buffer().get_image_data().get_data()))
        profiler.enable()
        simWindow.advance_frame(dt)
        profiler.disable()
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
    
    if button == 2:
        cell = solver.mesh.triFinder(x/WIDTH*pi,y/HEIGHT*pi)
        solver.density[solver.mesh.faces[cell]] = 1
        solver.vectors[solver.mesh.faces[cell]] = [0,3]
        for c in solver.mesh.faces[cell]:
            solver.source_cells.add(c)

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
        
        solver.vectors[solver.mesh.faces[cell]] += [
            simWindow.speed*dx, simWindow.speed*-dy
        ]

    if buttons == 2:
        cell = solver.mesh.triFinder(x/WIDTH*pi,y/HEIGHT*pi)
        solver.density[solver.mesh.faces[cell]] = 1
        solver.vectors[solver.mesh.faces[cell]] = [0,3]
        for c in solver.mesh.faces[cell]:
            solver.source_cells.add(c)
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
    import cv2
    import os
    from PIL import Image
    # run app
    import cProfile, pstats
    import matplotlib.pyplot as plt
    global profiler
    profiler = cProfile.Profile()
    # try:
    app.run()
    # except:
    def generate_triangular_surfaces(d):
        ax.clear()
        ax.set_xlim(0, pi)
        ax.set_ylim(0, pi)
        ax.set_zlim(0, 10)
        ax.plot_trisurf(solver.mesh.points[:,0], solver.mesh.points[:,1], abs(solver.div_history[d]), cmap=cm.coolwarm)
        
    # Attaching 3D axis to the figure
    my_dpi = 96
    plt.plot(abs(np.array(solver.div_history)).max(axis=1))
    plt.show(block=True)
    fig = plt.figure(figsize=(720/my_dpi, 720/my_dpi), dpi=my_dpi)
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(0, pi)
    ax.set_ylim(0, pi)
    ax.set_zlim(0, 10)
    anim = animation.FuncAnimation(fig, generate_triangular_surfaces, len(solver.div_history), interval=1000/60.0, repeat=False)
    progress_callback = lambda i, n: print(f'Saving frame {i} of {n}')
    anim.save('div_anim.mp4', fps=60, progress_callback=progress_callback)
    if simWindow.save_video:
        print('saving frames')
        video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 60, (WIDTH,HEIGHT))
        for f, frame  in tqdm(enumerate(frames)):
            if len(frame) != WIDTH*HEIGHT*4:
                break
            video.write(cv2.cvtColor(np.array(Image.frombuffer("RGBA", (WIDTH, HEIGHT), frame, "raw", "RGBA", 0, -1)), cv2.COLOR_RGBA2BGR))
        video.release()
#     print('acabou')
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()
