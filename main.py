from matplotlib import tri
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import cache
import string
import pyvista as pv
import numpy as np
from math import cos, pi
import scipy.spatial.distance as sd
import numpy.polynomial.polynomial as pp

def find_lambda(pr, face):
    A = [[1,1,1],
        [mesh.points[faces[face,0],0], mesh.points[faces[face,1],0], mesh.points[faces[face,2],0]],
        [mesh.points[faces[face,0],1], mesh.points[faces[face,1],1], mesh.points[faces[face,2],1]]]
    b = [1, pr[0], pr[1]]

    x = np.linalg.solve(A,b)

    return x

def interpolate(points, xy = None):
    cells = mesh.triFinder(points[:,0],points[:,1])
    lambdas = [find_lambda(x, y) for x,y in zip(points,cells)]
    if xy != None:
        return np.sum(mesh.vectors[:,xy][faces[cells]]*lambdas,axis=1)
    return np.sum(mesh.density[faces[cells]]*lambdas,axis=1)
    
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
    for pid in range(mesh.n_points):
        if is_boundary(mesh, boundary, pid):
            b_id.append(pid)
    return b_id

def get_normals(mesh, boundary_ids):
    normals = np.zeros((mesh.n_points,3))
    for id, b in zip(list(boundary_ids),mesh.points[list(boundary_ids)]):
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
    mesh['vectors'] = np.hstack((np.random.random((mesh.n_points,2)),np.zeros((mesh.n_points,1))))*2#np.zeros((mesh.n_points,3))#
    mesh['scalars'] = (np.linalg.norm(mesh['vectors'],axis=1))
    mesh.set_active_vectors('vectors')
    # mesh['density'][25] = 1
    # mesh['vectors'][25] = [0,1,0]

    return mesh

def computeExternalForces():
    pass

def computeViscosity():
    pass

def computePressure():
    x0 = poisson_solver()
    grad = gradient(x0)

    mesh.vectors[:,0] -= grad[:,0]
    mesh.vectors[:,1] -= grad[:,1]

    apply_boundary_condition()

def computeAdvection(density):
    new_pos = mesh.points-mesh.vectors*timeInterval
    new_pos = np.clip(new_pos,0,pi)
    if density:
        mesh.density = interpolate(new_pos)
    else:
        mesh.vectors[:,0] = interpolate(new_pos,0)
        mesh.vectors[:,1] = interpolate(new_pos,1)

    apply_boundary_condition()

def computeSource():
    mesh.density[[25,40]] = 1

def divergent(pid):
    div = np.sum(rbf[pid][:,1]*mesh.vectors[nring[pid],0] + rbf[pid][:,2]*mesh.vectors[nring[pid],1])
    apply_boundary_condition()
    return div
    

def gradient(lapl):
    grad = np.zeros((mesh.n_points,2))
    for pid in range(mesh.n_points):        
        grad[pid] = np.sum(rbf[pid][:,1]*lapl[nring[pid]]), np.sum(rbf[pid][:,2]*lapl[nring[pid]])
    return grad

def poisson_solver():
    lapl = np.asarray([])
    w = np.zeros(mesh.n_points)
    b = np.zeros(mesh.n_points)
    for pid in range(mesh.n_points): 
        if pid in boundary:
            weights = rbf[pid][:,1]*mesh.normals[pid,0] + rbf[pid][:,2]*mesh.normals[pid,1]
        else:
            weights = rbf[pid][:,0]
            b[pid] = divergent(pid)
        
        line = np.zeros(mesh.n_points)
        line[nring[pid]]= weights
        # p.show(cpos='xy')
        w = np.vstack([w, line])
    w = np.delete(w, 0,0)
    # w[0] = np.identity(mesh.n_points)[0]
    # b[0] = solution(mesh.points[0][0],mesh.points[0][1])
    np.savetxt("foo.csv", w, fmt="%.7s",delimiter="\t")

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

def update_field(*args):
    apply_boundary_condition()
    velocityStep()

    apply_boundary_condition()
    densityStep()

    im.set_array(mesh.density)
    vc.U = mesh.vectors[:,0]
    vc.V = mesh.vectors[:,1]

def pause_play(a):
    global _pause
    _pause = a

def stop_sim(a):
    global stop
    stop = a

def apply_boundary_condition():
    return
    mesh['vectors'][left_id,0] = 0
    mesh['vectors'][right_id,0] = 0
    mesh['vectors'][top_id,1] = 0
    mesh['vectors'][bottom_id,1] = 0

    # mesh['vectors'][40,1] = 1
    
def main():
    global faces    
    global mesh
    global timeInterval
    global boundary
    global pl
    global _pause
    global stop 
    global bottom_id 
    global top_id 
    global left_id
    global right_id
    global nring
    global rbf
    global im, vc

    timeInterval = 1/60.0

    mesh = pv.read("./mesh8.obj")
    assert mesh.is_all_triangles()

    mesh = tri.Triangulation(mesh.points[:,0], mesh.points[:,1])

    mesh.points = np.asarray([p for p in zip(mesh.x,mesh.y)])
    mesh.n_points = len(mesh.points)
    mesh.density = np.zeros((mesh.n_points))
    mesh.vectors = np.random.random((mesh.n_points,2))#np.zeros((mesh.n_points,2))
    mesh.triFinder = mesh.get_trifinder()
    faces = mesh.triangles
    # plt.style.use('dark_background')
    # plt.figure(figsize=(5,5),dpi=160)
    # plt.triplot(mesh)
    
    nring = [np.append(find_Nring(2, p, []),p) for p in range(len(mesh.x))]
    rbf = [rbf_fd_weights(mesh.points[nring[p]], mesh.points[p], 5, 2) for p in range(mesh.n_points)]
    # Find edges at the boundary
    boundary = set()
    for i in range(len(mesh.neighbors)):
        for k in range(3):
            if (mesh.neighbors[i][k] == -1):
                nk1,nk2 = (k)%3, (k)%3 
                boundary.add(mesh.triangles[i][nk1])
                boundary.add(mesh.triangles[i][nk2])

    mesh.normals = get_normals(mesh, boundary)
    bottom_id = [x for x in boundary if mesh.normals[x,1] < 0]
    top_id = [x for x in boundary if mesh.normals[x,1] > 0]
    left_id = [x for x in boundary if mesh.normals[x,0] < 0]
    right_id = [x for x in boundary if mesh.normals[x,0] > 0]

    px = [mesh.points[b][0] for b in boundary]
    py = [mesh.points[b][1] for b in boundary]
    # plt.scatter(px,py)


    fig, ax = plt.subplots()
    ax.set(xlim=(0,pi),ylim=(0,pi))
    #criando a figura e adiocionando os eixos

    # cid = fig.canvas.mpl_connect('button_press_event', on_button_press)
    # cid = fig.canvas.mpl_connect('button_release_event', on_button_release)
    # cid = fig.canvas.mpl_connect('motion_notify_event', on_motion)
    #mpl_connect mapeia para um inteiro ou eu que faço?
    # onde ele muda esse cid ao longo do código
    #im é a imagem que vamos trabalhar em cima, im.set_data() atualiza a im a cada chamada
    im = ax.tripcolor(mesh.points[:,0], mesh.points[:,1], np.random.random(mesh.n_points), shading='gouraud',cmap=plt.cm.gray)
    vc = plt.quiver(mesh.points[:,0],mesh.points[:,1],mesh.vectors[:,0],mesh.vectors[:,1],color='red')
    #[1:-1, 1:-1]: vai da primeira ate a ultima posiçao do array
    #extent: espaço que a imagem vai ocupar
    #vmin e vmax: range do color map
    animation = FuncAnimation(fig, update_field, interval=10, frames=800)
    #interval: delay entre os frames em milissegundos, super rápido
    plt.show()
    

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