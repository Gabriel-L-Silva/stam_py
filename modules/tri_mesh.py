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
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from libpysal.weights import KNN


class TriMesh:
    def __init__(self, filename, k=12, s=5, d=2, only_knn=False):
        self.mesh = trimesh.load_mesh(filename)

        self.k = k
        self.s = s
        self.d = d
        self.only_knn = only_knn
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
        knn = list(KNN.from_array(self.points, k=self.k).neighbors.values())
        if self.only_knn: 
            self.nring = [np.append(knn[p],[p]) for p in tqdm(range(len(self.mesh.vertices)))]
        else:
            self.nring = [list(self.find_Nring(2, p, set())) for p in tqdm(range(len(self.mesh.vertices)))]
            for id, ring in enumerate(self.nring):
                if len(ring) < self.k:
                    ring = knn[id].append(id)
        print('Building rbf...')
        self.rbf = [rbf_fd_weights(self.points[self.nring[p]], self.points[p], self.s, self.d) for p in tqdm(range(self.n_points))]
        
        self._init_boundary()

    def _init_boundary(self):
        # Find edges at the boundary
        unique_edges = self.mesh.edges[trimesh.grouping.group_rows(self.mesh.edges_sorted, require_count=1)]
        self.boundary = set(np.unique(unique_edges.flatten()))

        self.normals = self._get_normals(unique_edges)

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
        return e

    def _get_normals(self, edges):
        normals = np.zeros((self.n_points,3))
        e_normals = np.zeros((len(edges),2))
        R_matrix = np.array([[0,-1],[1,0]])
        sorted_edges = self.sort_edges(edges)
        for idx, edge in enumerate(sorted_edges):
            e = self.mesh.vertices[edge[0]][:2] - self.mesh.vertices[edge[1]][:2]
            e_normals[idx] = R_matrix@e
            e_normals[idx] = e_normals[idx] / np.sqrt(np.sum(e_normals[idx]**2))
        for id, edge in enumerate(sorted_edges):
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
        
        # import matplotlib.pyplot as plt
        # plt.plot(self.points[edges,0], self.points[edges,1], 'o')
        # plt.quiver(self.points[edges,0], self.points[edges,1], normals[edges,0], normals[edges,1])
        # plt.show(block=True)
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