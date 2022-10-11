from matplotlib import tri
import networkx as nx
from functools import cache
from math import pi
import trimesh
import numpy as np
from tqdm import tqdm
try:
    from modules.rbf import rbf_fd_weights
except:
    from rbf import rbf_fd_weights

class TriMesh:
    def __init__(self, filename):
        self.mesh = trimesh.load_mesh(filename)

        self._init_mesh()

    
    def triFinder(self, x, y):
        cells = self.findTri(x,y)
        if cells.min() == -1:
            if(type(x) == float):
                points= np.array([[x, y,0]])
            else:
                points = np.stack([x,y,np.zeros(len(x))]).T
            _,_,cells = trimesh.proximity.closest_point(self.mesh, points)
        return cells

    def _init_mesh(self):
        self.t_mesh = tri.Triangulation(self.mesh.vertices[:,0], self.mesh.vertices[:,1], self.mesh.faces)

        self.points = np.asarray([p for p in zip(self.mesh.vertices[:,0],self.mesh.vertices[:,1])])
        self.n_points = len(self.points)

        self.findTri = self.t_mesh.get_trifinder()
        self.faces = self.mesh.faces
        
        print('Building neighborhood...')
        self.g = nx.from_edgelist(self.mesh.edges_unique)
        self.nring = [list(self.find_Nring(2, p, set())) for p in tqdm(range(len(self.mesh.vertices)))]
        print('Building rbf...')
        self.rbf = [rbf_fd_weights(self.points[self.nring[p]], self.points[p], 5, 2) for p in tqdm(range(self.n_points))]
        
        self._init_boundary()

    def _init_boundary(self):
        # Find edges at the boundary
        unique_edges = self.mesh.edges[trimesh.grouping.group_rows(self.mesh.edges_sorted, require_count=1)]
        self.boundary = set(np.unique(unique_edges.flatten()))

        self.normals = self._get_normals()

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

    def find_one_ring(self, index):
        return list(self.g[index].keys())
    
    def find_Nring(self, n, index, nh :set):
        if n == 0:
            nh.add(index)
            return
        else:
            for v in self.find_one_ring(index):
                self.find_Nring(n-1, v, nh)
            return nh