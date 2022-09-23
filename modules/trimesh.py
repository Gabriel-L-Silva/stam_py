from functools import cache
from math import pi
import pyvista as pv
import numpy as np
from matplotlib import tri
try:
    from modules.rbf import rbf_fd_weights
except:
    from rbf import rbf_fd_weights

class TriMesh:
    def __init__(self, filename):
        mesh = pv.read(filename)
        assert mesh.is_all_triangles()

        self._init_mesh(mesh)

    def _init_mesh(self, mesh):
        self.mesh = tri.Triangulation(mesh.points[:,0], mesh.points[:,1])

        self.points = np.asarray([p for p in zip(self.mesh.x,self.mesh.y)])
        self.n_points = len(self.points)

        self.triFinder = self.mesh.get_trifinder()
        self.faces = self.mesh.triangles
        
        self.nring = [np.append(self.find_Nring(2, p, []),p) for p in range(self.n_points)]
        self.rbf = [rbf_fd_weights(self.points[self.nring[p]], self.points[p], 5, 2) for p in range(self.n_points)]
        
        self._init_boundary()

    def _init_boundary(self):
        # Find edges at the boundary
        self.boundary = set()
        for i in range(len(self.mesh.neighbors)):
            for k in range(3):
                if (self.mesh.neighbors[i][k] == -1):
                    nk1,nk2 = (k)%3, (k)%3 
                    self.boundary.add(self.faces[i][nk1])
                    self.boundary.add(self.faces[i][nk2])

        self.normals = self._get_normals()

        self.bottom_id = [x for x in self.boundary if self.normals[x,1] < 0]
        self.top_id = [x for x in self.boundary if self.normals[x,1] > 0]
        self.left_id = [x for x in self.boundary if self.normals[x,0] < 0]
        self.right_id = [x for x in self.boundary if self.normals[x,0] > 0]

    def _get_normals(self):
        normals = np.zeros((self.n_points,3))
        for id, b in zip(list(self.boundary),self.points[list(self.boundary)]):
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

    @cache
    def find_faces_with_node(self, index):
        """Pass the index of the node in question.
        Returns the face indices of the faces with that node."""
        return [i for i, face in enumerate(self.faces) if index in face]
    @cache
    def find_connected_vertices(self, index):
        """Pass the index of the node in question.
        Returns the vertex indices of the vertices connected with that node."""
        cids = self.find_faces_with_node(index)
        connected = np.unique(self.faces[cids].ravel())
        return np.delete(connected, np.argwhere(connected == index))

    def find_Nring(self, n, index, nh = []):
        if n == 0:
            nh.append(index)
            return
        else:
            for v in self.find_connected_vertices(index):
                self.find_Nring(n-1, v, nh)
            connected = np.unique(nh)
            return np.delete(connected, np.argwhere(connected == index))