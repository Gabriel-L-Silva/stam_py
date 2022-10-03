import numpy as np

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
        if cells.min() == -1:
            print('a')
        lambdas = [self.find_lambda(x, y) for x,y in zip(points,cells)]

        return np.sum(data[self.mesh.faces[cells]]*lambdas,axis=1)
    
def func(x,y):
    return 2*x+y**2

def main(): 
    from matplotlib import tri
    import matplotlib.pyplot as plt
    from matplotlib import cm

    from tri_mesh import TriMesh
    mesh = TriMesh("./assets/mesh16.obj")
    xy = np.random.rand(mesh.n_points, 2)*np.pi
    points = np.stack((xy[:,0],xy[:,1],np.zeros(mesh.n_points)),axis=1)

    solution = func(points[:,0], points[:,1])

    Interp = Interpolator(mesh)
    CInterp = tri.CubicTriInterpolator(mesh.mesh, func(mesh.points[:,0],mesh.points[:,1]), kind='min_E')
    interp = Interp(func(mesh.points[:,0],mesh.points[:,1]), points[:,:2])
    Cinterp = CInterp(points[:,0],points[:,1])

    error = (interp-solution)
    Cerror = (Cinterp-solution)
    
    print(max(error))

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')

    # Plot the surface.
    ax1.set_title('Nossa solução')
    ax2.set_title('Solução Exata')
    ax3.set_title('Erro')
    ax3.set_title('Erro cubico')
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