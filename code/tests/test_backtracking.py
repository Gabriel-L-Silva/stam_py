import pytest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import sys
from shapely import Point

def on_press(event):
    print('press', event.key)
    sys.stdout.flush()
    if event.key == 'x':
        visible = o.get_visible()
        o.set_visible(not visible)
        visible = q.get_visible()
        q.set_visible(not visible)
        visible = fp.get_visible()
        fp.set_visible(not visible)
        visible = corr.get_visible()
        corr.set_visible(not visible)

        fig.canvas.draw()

@pytest.mark.parametrize("solver, custom_intersection", [['quad_solver', True], ['circle_solver',False], ['circle_solver',True]])
# @pytest.mark.parametrize("solver", ['circle_solver'])
def test_backtracking(solver, custom_intersection, request):
    global o, q, fp, corr, fig
    fig, ax = plt.subplots()

    fig.canvas.mpl_connect('key_press_event', on_press)
    solver = request.getfixturevalue(solver)

    solver.vectors[:,:2] = (np.random.random((solver.mesh.mesh.vertices.shape[0], 2))*2 - 1)*2
    
    dt = 1/60.0
    new_pos = solver.backtrace(dt)

    out_of_bounds_idx = np.where([not solver.mesh.boundary.contains(Point(p)) for p in new_pos])[0][-10:]

    ax.plot(*solver.mesh.boundary.exterior.xy, c='k')
    o = ax.scatter(solver.mesh.mesh.vertices[out_of_bounds_idx,0], solver.mesh.mesh.vertices[out_of_bounds_idx,1], color='b')
    q = ax.quiver(solver.mesh.mesh.vertices[out_of_bounds_idx,0], solver.mesh.mesh.vertices[out_of_bounds_idx,1], -solver.vectors[out_of_bounds_idx,0]*dt, -solver.vectors[out_of_bounds_idx,1]*dt, color='r', angles='xy', scale_units='xy', scale=1)
    ax.scatter(new_pos[out_of_bounds_idx,0], new_pos[out_of_bounds_idx,1], color='r')

    intersected_pos = solver.intersect_boundary(new_pos, custom_intersection)
    correction = intersected_pos-new_pos
    fp = ax.scatter(intersected_pos[out_of_bounds_idx,0], intersected_pos[out_of_bounds_idx,1], color='g')
    corr = ax.quiver(new_pos[out_of_bounds_idx,0], new_pos[out_of_bounds_idx,1], correction[out_of_bounds_idx,0], correction[out_of_bounds_idx,1], color='g', angles='xy', scale_units='xy', scale=1)
    fp.set_visible(False)
    corr.set_visible(False)
    plt.show()

    # out_of_bounds_idx = np.where([not solver.mesh.boundary.contains(Point(p)) for p in intersected_pos])[0][-10:]
    # assert len(out_of_bounds_idx) == 0



