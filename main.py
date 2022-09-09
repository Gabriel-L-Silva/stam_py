from functools import cache
import string
from xml.etree.ElementInclude import include
import pyvista as pv
import numpy as np
from math import cos, pi
import scipy.spatial.distance as sd
import numpy.polynomial.polynomial as pp
# from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import pyvistaqt as pvqt
import random
from threading import Thread
import sys
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget)
import imageio

def find_lambda(pr, face):
    A = [[1,1,1],
        [mesh.points[faces[face,0],0], mesh.points[faces[face,1],0], mesh.points[faces[face,2],0]],
        [mesh.points[faces[face,0],1], mesh.points[faces[face,1],1], mesh.points[faces[face,2],1]]]
    b = [1, pr[0], pr[1]]

    x = np.linalg.solve(A,b)

    return x

def interpolate(points, field:string, xy = None):
    cells = mesh.find_containing_cell(points)
    lambdas = [find_lambda(x, y) for x,y in zip(points,cells)]
    if xy != None:
        return np.sum(mesh[field][faces[cells]][:,xy]*lambdas,axis=1)
    return np.sum(mesh[field][faces[cells]]*lambdas,axis=1)

class MouseTracker(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setMouseTracking(True)

    def initUI(self):
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Mouse Tracker')
        self.label = QLabel(self)
        self.label.resize(200, 40)
        self.show()

    def mouseMoveEvent(self, event):
        self.label.setText('Mouse coords: ( %d : %d )' % (event.x(), event.y()))
    
def rbf_fd_weights(X,ctr,s,d):
    #   X : each row contains one node in R^2
    # ctr : center (evaluation) node
    # s,d : PHS order and polynomial degree
        
    rbf  = lambda r,s: r**s
    Drbf = lambda r,s,xi: s*xi*r**(s-2)
    Lrbf = lambda r,s: s**2*r**(s-2)
        
    n = X.shape[0] 
    for i in range(2): X[:,i] -= ctr[i]
    DM = sd.squareform(sd.pdist(X))
    D0 = np.sqrt(X[:,0]**2 + X[:,1]**2)
    A = rbf(DM,s)
    b = np.vstack((Lrbf(D0,s),Drbf(D0,s,-X[:,0]),
                    Drbf(D0,s,-X[:,1]))).T
                    
    if d > -1: #adding polynomials
        m = int((d+2)*(d+1)/2)              
        O, k = np.zeros((m,m)), 0
        P, LP = np.zeros((n,m)), np.zeros((m,3))
        PX = pp.polyvander(X[:,0],d)
        PY = pp.polyvander(X[:,1],d)
        for j in range(d+1):
            P[:,k:k+j+1], k = PX[:,j::-1]*PY[:,:j+1], k+j+1
        if d > 0: LP[1,1], LP[2,2] = 1, 1
        if d > 1: LP[3,0], LP[5,0] = 2, 2 
        A = np.block([[A,P],[P.T,O]])	
        b = np.block([[b],[LP]])
    # each column contains the weights for 
    # the Laplacian, d/dx1, d/dx2, respectivly.
    weights = np.linalg.solve(A,b)
    return weights[:n,:]

def is_boundary(mesh, boundary, pid):
    return np.asarray([(mesh.points[pid] == p).all() for p in boundary.points]).any()

def get_boundary_ids(mesh, boundary):
    b_id = []
    for pid in range(mesh.number_of_points):
        if is_boundary(mesh, boundary, pid):
            b_id.append(pid)
    return b_id

def get_normals(mesh, boundary_ids):
    normals = np.zeros((mesh.number_of_points,3))
    for id, b in zip(boundary_ids,mesh.points[boundary_ids]):
        if b[0] == 0:
            normals[id] += [-1,0,0]
        if b[1] == 0:
            normals[id] += [0,-1,0]
        if abs(b[0] - pi) <= 10e-3:
            normals[id] += [1,0,0]
        if abs(b[1] - pi) <= 10e-3:
            normals[id] += [0,1,0]
        normals[id] = normals[id] / np.sqrt(np.sum(normals[id]**2))
    return normals

def initialize_mesh():
    mesh = pv.read("./mesh8.obj")
    assert mesh.is_all_triangles()

    mesh.point_data['density'] = np.zeros((mesh.n_points))
    mesh['colors'] = np.hstack((np.ones((mesh.n_points,3)),mesh['density'].reshape(mesh.n_points,1)))
    mesh['vectors'] = np.hstack((np.random.random((mesh.n_points,2)),np.zeros((mesh.n_points,1))))*2
    mesh['scalars'] = (np.linalg.norm(mesh['vectors'],axis=1))
    mesh['density'][36:40] = 1

    return mesh

def computeExternalForces():
    pass

def computeViscosity():
    pass

def computePressure():
    x0 = poisson_solver()
    grad = gradient(x0)

    mesh['vectors'][:,0] -= grad[:,0]
    mesh['vectors'][:,1] -= grad[:,1]

    apply_boundary_condition()

def computeAdvection(density):
    new_pos = mesh.points-mesh['vectors']*timeInterval
    new_pos = np.clip(new_pos,0,pi)
    if density:
        mesh.set_active_scalars('density')
        mesh['density'] = interpolate(new_pos,'density').reshape((-1,1))
    else:
        mesh['vectors'][:,0] = interpolate(new_pos,'vectors', 0)
        mesh['vectors'][:,1] = interpolate(new_pos,'vectors', 1)

    apply_boundary_condition()

def computeSource():
    mesh['density'][36:40] = 1

def divergent(rbf, nring):
    div = np.sum(rbf[:,1]*mesh['vectors'][nring,0] + rbf[:,2]*mesh['vectors'][nring,1])
    apply_boundary_condition()
    return div
    

def gradient(lapl):
    grad = np.zeros((mesh.n_points,2))
    for pid in range(mesh.number_of_points):
        nring = find_Nring(2, pid, [])
        nring = np.append(nring, nring[0])
        nring[0] = pid
        
        
        rbf = rbf_fd_weights(np.asarray([mesh.points[x] for x in nring]), np.asarray(mesh.points[pid]), 5, 2)
        
        grad[pid] = np.sum(rbf[:,1]*mesh['vectors'][nring,0]), np.sum(rbf[:,2]*mesh['vectors'][nring,1])
    return grad

def poisson_solver():
    lapl = np.asarray([])
    w = np.zeros(mesh.number_of_points)
    b = np.zeros(mesh.number_of_points)
    for pid in range(mesh.number_of_points):
        nring = find_Nring(2, pid, [])
        nring = np.append(nring, nring[0])
        nring[0] = pid
        
        weights = np.zeros_like(nring)
        rbf = rbf_fd_weights(np.asarray([mesh.points[x] for x in nring]), np.asarray(mesh.points[pid]), 5, 2)
        if pid in boundary_ids:
            weights = rbf[:,1]*mesh['normals'][pid,0] + rbf[:,2]*mesh['normals'][pid,1]
        else:
            weights = rbf[:,0]
            b[pid] = divergent(rbf, nring)
        
        line = np.zeros(mesh.number_of_points)
        line[nring]= weights
        # p.show(cpos='xy')
        w = np.vstack([w, line])
    w = np.delete(w, 0,0)
    # w[0] = np.identity(mesh.number_of_points)[0]
    # b[0] = solution(mesh.points[0][0],mesh.points[0][1])
    lapl = np.linalg.solve(w,b)
    return lapl

def velocityStep():
    computeExternalForces()

    computeViscosity()

    computePressure()

    computeAdvection(False)

    # computePressure()

def densityStep():
    computeSource()

    computeViscosity()

    computeAdvection(True)

def update_field():
    apply_boundary_condition()
    velocityStep()

    apply_boundary_condition()
    densityStep()

def pause_play(a):
    global _pause
    _pause = a

def stop_sim(a):
    global stop
    stop = a

def apply_boundary_condition():
    bottom_id = [x for x in boundary_ids if mesh['normals'][x,1] < 0]
    top_id = [x for x in boundary_ids if mesh['normals'][x,1] > 0]
    left_id = [x for x in boundary_ids if mesh['normals'][x,0] < 0]
    right_id = [x for x in boundary_ids if mesh['normals'][x,0] > 0]
    mesh['vectors'][left_id,0] = 0
    mesh['vectors'][right_id,0] = 0
    mesh['vectors'][top_id,1] = 0
    mesh['vectors'][bottom_id,1] = 0
    
def main():
    global faces    
    global mesh
    global timeInterval
    global boundary_ids
    global pl
    global _pause
    global stop 

    stop = True
    _pause = True
    timeInterval = 1/60.0
    #init mesh
    mesh = initialize_mesh()

    filename = 'smoke_sim.mp4'
    #create window
    pl = pvqt.BackgroundPlotter()
    pl.open_movie(filename,framerate=60)
    # pl.view_xy()
    pl.camera_position = [(1.6690598396129241, 1.3942212637901101, 8.140274938683984),
        (1.6690598396129241, 1.3942212637901101, 0.0),
        (0.0, 1.0, 0.0)
    ]

    faces = mesh.faces.reshape((-1,4))[:, 1:4]

    #calculate boundary normals
    boundary = mesh.extract_feature_edges(boundary_edges=True, 
                                        non_manifold_edges=False, 
                                        manifold_edges=False)
    boundary_ids = get_boundary_ids(mesh, boundary)
    mesh['normals'] = get_normals(mesh, boundary_ids)
    apply_boundary_condition()
    #show vectors
    # geom = pv.Arrow()
    # glyphs = mesh.glyph(orient="vectors", scale="scalars", factor=0.1, geom=geom)
    # vec_actor = pl.add_mesh(glyphs, show_scalar_bar=False, lighting=False, cmap='coolwarm',)
    pl.add_mesh(mesh, show_edges=True, scalars='colors', rgba=True)
    pl.add_checkbox_button_widget(pause_play, position=(10, 150),value=True)
    pl.add_checkbox_button_widget(stop_sim, position=(0, 150),value=True,color='red')
    # pl.add_callback(update_field, interval=int(timeInterval))
    while stop:
        if _pause:
            pl.app.processEvents()
            continue
        # update velocity
        # mesh['colors'] = np.hstack((np.ones((mesh.n_points,3)),mesh['density'].reshape(mesh.n_points,1)))
        # mesh.set_active_scalars('scalars')
        # glyphs = mesh.glyph(orient="vectors", scale="scalars", factor=0.1, geom=geom)
        # pl.remove_actor(vec_actor)
        
        # vec_actor = pl.add_mesh(glyphs, show_scalar_bar=False, lighting=False, cmap='coolwarm',render=False)
        # mesh.set_active_scalars('density')
        update_field()
        pl.render()
        pl.write_frame()
        pl.app.processEvents()
    

@cache
def find_faces_with_node(index):
    """Pass the index of the node in question.
    Returns the face indices of the faces with that node."""
    return [i for i, face in enumerate(faces) if index in face]
@cache
def find_connected_vertices(index):
    """Pass the index of the node in question.
    Returns the vertex indices of the vertices connected with that node."""
    cids = find_faces_with_node(index)
    connected = np.unique(faces[cids].ravel())
    return np.delete(connected, np.argwhere(connected == index))

def find_Nring(n, index, nh = []):
    if n == 0:
        nh.append(index)
        return
    else:
        for v in find_connected_vertices(index):
            find_Nring(n-1, v, nh)
        connected = np.unique(nh)
        return np.delete(connected, np.argwhere(connected == index))

if __name__ == '__main__':
    main()