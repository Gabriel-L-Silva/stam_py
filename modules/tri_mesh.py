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
    def __init__(self, mesh):
        self.mesh = mesh

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

    def _init_nring(self):
        self.nring = np.zeros((self.n_points,self.n_points), dtype = int)
        for p in tqdm(range(len(self.mesh.vertices))):
            self.nring[p, list(self.find_Nring(2, p, set()))] = 1

    def _init_rbf(self):
        self.rbf = np.zeros((self.n_points, self.n_points, 3))
        for p in tqdm(range(len(self.mesh.vertices))):
            self.rbf[p, np.argwhere(self.nring[p]).flatten()] = rbf_fd_weights(self.points[np.argwhere(self.nring[p]).flatten()], self.points[p], 5, 2) 

    def _init_mesh(self):
        self.t_mesh = tri.Triangulation(self.mesh.vertices[:,0], self.mesh.vertices[:,1], self.mesh.faces)
        self.mesh.vertices = np.stack((self.mesh.vertices[:,0],self.mesh.vertices[:,1], np.zeros(len(self.mesh.vertices))),axis=1)
        self.points = np.asarray([p for p in zip(self.mesh.vertices[:,0],self.mesh.vertices[:,1])])
        self.n_points = len(self.points)
        self.faces = self.mesh.faces

        self.findTri = self.t_mesh.get_trifinder()
        
        print('Building neighborhood...')
        self.g = nx.from_edgelist(self.mesh.edges_unique)
        self._init_nring()
        print('Building rbf...')
        self._init_rbf()
        
        self._init_boundary()

    def _init_boundary(self):
        # Find edges at the boundary
        unique_edges = self.mesh.edges[trimesh.grouping.group_rows(self.mesh.edges_sorted, require_count=1)]
        self.boundary = set(np.unique(unique_edges.flatten()))

        self.normals = self._get_normals(unique_edges)

    def sort_edges(self, EDGES):
        sorted_edges = []
        edges = [list(e) for e in EDGES]
        from collections import defaultdict

        followed_by = defaultdict(list)

        def follows(edge1, edge2):  # does edge2 follow edge1
            return edge1 != edge2 and not set(edge1).isdisjoint(set(edge2))

        def sorted_path(path, end):

            start = path[-1]

            for follower in followed_by[tuple(start)]:

                if follower in path:
                    continue  # avoid circularities

                if follower == end:
                    return path + [end]  # solution found

                new_path = sorted_path(path + [follower], end)  # recurse

                if new_path:
                    return new_path  # solution found

            return None  # solution not found

        # build defaultdict of who follows who

        for edge in edges:
            for potential_follower in edges:
                if follows(edge, potential_follower):
                    followed_by[tuple(edge)].append(potential_follower)

        return np.array(sorted_path([edges[0]], edges[np.where(EDGES[:,1] == 0)[0][0]]))

    def _get_normals(self, edges):
        normals = np.zeros((self.n_points,3))
        e_normals = np.zeros((len(edges),2))
        R_matrix = np.array([[0,1],[-1,0]])
        sorted_edges = self.sort_edges(edges)
        for idx, edge in enumerate(sorted_edges):
            e = self.mesh.vertices[edge[0]][:2] - self.mesh.vertices[edge[1]][:2]
            e_normals[idx] = R_matrix@e
            e_normals[idx] = e_normals[idx] / np.sqrt(np.sum(e_normals[idx]**2))
        for id, edge in enumerate(sorted_edges):
            if id==131:
                print('aq')
            normals[edge[0],:2] = (e_normals[id-1] + e_normals[id])/2
            normals[edge[0],2] = 0
            normals[edge[0]] = normals[edge[0]] / np.sqrt(np.sum(normals[edge[0]]**2))
        # for id, b in zip(list(self.boundary),self.points[list(self.boundary)]):
        #     if b[0] == 0:
        #         normals[id] += [-1,0,0]
        #     if b[1] == 0:
        #         normals[id] += [0,-1,0]
        #     if abs(b[0] - pi) <= 10e-3:
        #         normals[id] += [1,0,0]
        #     if abs(b[1] - pi) <= 10e-3:
        #         normals[id] += [0,1,0]
        #     normals[id] = normals[id] / np.sqrt(np.sum(normals[id]**2))

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

if __name__ == '__main__':
    TriMesh('./assets/mesh16.obj')