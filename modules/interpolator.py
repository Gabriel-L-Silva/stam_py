import numpy as np
from tri_mesh import TriMesh
from rbf import rbf_interpolator_inv_matrix, rbf 
import scipy.spatial.distance as sd
from tqdm import tqdm

class Interpolator:
    def __init__(self, mesh) -> None:
        self.mesh = mesh

    def find_lambda(self, pr, face):
        A = [[1,1,1],
            [self.mesh.points[self.mesh.faces[face,0],0], self.mesh.points[self.mesh.faces[face,1],0], self.mesh.points[self.mesh.faces[face,2],0]],
            [self.mesh.points[self.mesh.faces[face,0],1], self.mesh.points[self.mesh.faces[face,1],1], self.mesh.points[self.mesh.faces[face,2],1]]]
        b = [1, pr[0], pr[1]]

        x = np.linalg.solve(A,b)

        return x

    def __call__(self, data, points):
        cells = self.mesh.triFinder(points[:,0],points[:,1])
        lambdas = [self.find_lambda(x, y) for x,y in zip(points,cells)]

        return np.sum(data[self.mesh.faces[cells]]*lambdas,axis=1)

class RBFInterpolator:
    def __init__(self, mesh: TriMesh) -> None:
        self.mesh = mesh
        self.nring = []
        for f in mesh.faces:
            nring = set()
            nb = [mesh.find_one_ring(v) for v in f]
            for n in nb:
                for c in n:
                    nring.add(c)
            self.nring.append(list(nring))
        self.nring = np.asarray(self.nring,dtype=object)
        self.rbf = []
        self.poly = []
        for f in tqdm(range(len(mesh.faces))):
            ret = rbf_interpolator_inv_matrix(mesh.points[self.nring[f]], 5, 2, np.zeros((len(self.nring[f]),2)), np.zeros(len(self.nring[f])))
            self.rbf.append(ret[0])
            self.poly.append(ret[1])

    def __call__(self, data, points):
        cells = self.mesh.triFinder(points[:,0],points[:,1])
        value_interp = np.zeros(points.shape[0])
        for idx, c in enumerate(cells):
            pad_size = len(self.rbf[c])-len(data[self.nring[c]])
            alphas = np.dot(self.rbf[c], np.pad(data[self.nring[c]],(0,pad_size)))
            value_interp[idx] = (np.sum(alphas[:-pad_size]*rbf(sd.cdist(points,self.mesh.points[self.nring[c]]),5) + np.sum(alphas[-pad_size:]*self.poly[c])))
        return value_interp

def func(x,y):
    return np.sin(2*x+y**2)+np.cos(x*y-2*x**2)

def rmse(e, N):
    return np.sum(np.sqrt(e**2/N))

def main(): 
    from matplotlib import tri
    import matplotlib.pyplot as plt
    from matplotlib import cm

    from tri_mesh import TriMesh
    N = 64
    mesh = TriMesh(f"./assets/regular_tri_grid{N}.obj")
    xy = np.random.rand(mesh.n_points, 2)*np.pi
    points = np.stack((xy[:,0],xy[:,1],np.zeros(mesh.n_points)),axis=1)

    solution = func(points[:,0], points[:,1])

    Interp = Interpolator(mesh)
    CInterp = tri.CubicTriInterpolator(mesh.t_mesh, func(mesh.points[:,0],mesh.points[:,1]), kind='min_E')
    RBFInterp = RBFInterpolator(mesh)
    interp = Interp(func(mesh.points[:,0],mesh.points[:,1]), points[:,:2])
    rbfinterp = RBFInterp(func(mesh.points[:,0],mesh.points[:,1]), points[:,:2])
    Cinterp = CInterp(points[:,0],points[:,1])

    error = (interp-solution)
    Cerror = (Cinterp-solution)
    
    print(f'max: {max(abs(error))}, rmse: {rmse(error, N)}')
    print(f'Cmax: {max(abs(Cerror))}, Crmse: {rmse(Cerror, N)}')
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')

    # Plot the surface.
    ax1.set_title('Nossa solução')
    ax2.set_title('Solução Exata')
    ax3.set_title('Erro')
    ax4.set_title('Erro cubico')
    # Plot the surface.
    surf = ax1.plot_trisurf(points[:,0], points[:,1], interp, cmap=cm.coolwarm                        )
    surf = ax2.plot_trisurf(points[:,0], points[:,1], solution, cmap=cm.coolwarm                       )
    surf3 = ax3.plot_trisurf(points[:,0], points[:,1], error, cmap=cm.coolwarm                        )
    surf4 = ax4.plot_trisurf(points[:,0], points[:,1], Cerror, cmap=cm.coolwarm                        )
    fig.colorbar(surf3, ax=ax3, fraction=0.1, pad=0.2)
    fig.colorbar(surf4, ax=ax4, fraction=0.1, pad=0.2)
    fig.suptitle(f'Norma infito do erro = {max(error):1e}', fontsize=20)
    plt.show(block=True)

if __name__ == '__main__':
    main()