import numpy as np

class Interpolator:
    def __init__(self, mesh, data) -> None:
        self.mesh = mesh
        self.data = data

    def find_lambda(self, pr, face):
        A = [[1,1,1],
            [self.mesh.points[self.mesh.faces[face,0],0], self.mesh.points[self.mesh.faces[face,1],0], self.mesh.points[self.mesh.faces[face,2],0]],
            [self.mesh.points[self.mesh.faces[face,0],1], self.mesh.points[self.mesh.faces[face,1],1], self.mesh.points[self.mesh.faces[face,2],1]]]
        b = [1, pr[0], pr[1]]

        x = np.linalg.solve(A,b)

        return x

    def __call__(self, points):
        cells = self.mesh.triFinder(points[:,0],points[:,1])
        lambdas = [self.find_lambda(x, y) for x,y in zip(points,cells)]

        return np.sum(self.data[self.mesh.faces[cells]]*lambdas,axis=1)