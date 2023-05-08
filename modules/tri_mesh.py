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
from libpysal.weights import KNN

import matplotlib.pyplot as plt
class TriMesh:
    def __init__(self, mesh, k=12):
        self.mesh = mesh

        self.k = k
        self._init_mesh()

    def triFinder(self, x, y):
        cells = self.findTri(x,y)
        if cells.min() == -1:
            if(type(x) == float or type(x) == np.float64):
                points= np.array([[x, y,0]])
            else:
                points = np.stack([x,y,np.zeros(len(x))]).T
            _,_,cells = trimesh.proximity.closest_point(self.mesh, points)
        return cells

    def _init_nring(self):
        self.nring = np.zeros((self.n_points,self.n_points), dtype = int)
        knn = list(KNN.from_array(self.points, k=self.k).neighbors.values())
        for p in tqdm(range(len(self.mesh.vertices))):
            self.nring[p, list(self.find_Nring(2, p, set()))] = 1
            if np.count_nonzero(self.nring[p]) < 12:
                self.nring[p, knn[p]] = 1

    def _init_rbf(self):
        self.rbf = np.zeros((self.n_points, self.n_points, 3))
        for p in tqdm(range(len(self.mesh.vertices))):
            self.rbf[p, np.argwhere(self.nring[p]).flatten()] = rbf_fd_weights(self.points[np.argwhere(self.nring[p]).flatten()], self.points[p], 5, 2) 

    def _init_mesh(self):
        self.t_mesh = tri.Triangulation(self.mesh.vertices[:,0], self.mesh.vertices[:,1], self.mesh.faces)
        self.mesh.vertices = np.stack((self.mesh.vertices[:,0],self.mesh.vertices[:,1], np.zeros(len(self.mesh.vertices))),axis=1)
        self.points = np.array([p for p in zip(self.mesh.vertices[:,0],self.mesh.vertices[:,1])])
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
        self.boundary_set = set(np.unique(unique_edges.flatten()))
        self.boundary = self.mesh.boundary
        plt.scatter(self.mesh.vertices[list(self.boundary_set),0], self.mesh.vertices[list(self.boundary_set),1], c='r')
        plt.show(block=True)
        self.normals = self._get_normals(self.sort_edges(unique_edges))

    def sort_edges(self, edges):
        edge_count = len(edges)
        if(edge_count > 0):
            e = []
            e.append(edges[0])
            flip = []
            
            #main loop for edge e
            for e_iter in edges:
                eb = e[-1][1] #end vertex of last edge of e
                found = False
                
                #first search
                for m in edges:
                    ma = m[0] #begin vertex of edge m
                    mb = m[1] #end vertex of edge m
                    if(eb == ma):
                        flipFound = False
                        for n in e:
                            if(n[0] == mb and n[1] == ma):
                                flipFound = True
                                break
                        if(flipFound == True):
                            continue
                        else:         
                            e.append(m)
                            found = True
                            break
                    
                #check for reverse direction in case first search failed
                if(found == False):
                    for index, m in enumerate(edges):
                        ma = m[0] #begin vertex of edge e
                        mb = m[1] #end vertex of edge e
                        
                        #...also exclude existing m's in e
                        if(mb == eb and m not in e):
                            #create duplicate to reverse vertex indices
                            m_dup = edges.copy()
                            f = m_dup[index]
                            f[0] = mb
                            f[1] = ma
                            e.append(f)
                        else:
                            continue
    
        #remove last element (was added twice)
        del e[-1]
        return np.array(e)

    def _get_normals(self, edges):
        normals = np.zeros((self.n_points,3))
        e_normals = np.zeros((len(edges),2))
        R_matrix = np.array([[0,-1],[1,0]])
        sorted_edges = self.sort_edges(edges)
        self.sorted_edges = sorted_edges
        for idx, edge in enumerate(sorted_edges):
            e = self.mesh.vertices[edge[0]][:2] - self.mesh.vertices[edge[1]][:2]
            e_normals[idx] = R_matrix@e
        e_normals /= np.linalg.norm(e_normals, axis=1)[:,None]
        for id, edge in enumerate(sorted_edges):
            normals[edge[0],:2] = (e_normals[id-1] + e_normals[id])/2
            normals[edge[0],2] = 0
        normals /= np.linalg.norm(normals, axis=1)[:,None]

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(self.mesh.vertices[edges,0], self.mesh.vertices[edges,1], 'o')
        # ax.quiver(self.mesh.vertices[edges,0], self.mesh.vertices[edges,1], normals[edges,0], normals[edges,1])
        # plt.show()

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