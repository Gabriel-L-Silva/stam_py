from math import pi
import numpy as np
try:
    from modules.trimesh import TriMesh
except:
    from trimesh import TriMesh
try:
    from modules.interpolator import Interpolator
except:
    from interpolator import Interpolator

class TriSolver:
    def __init__(self, filename, WIDTH, HEIGHT):
        self.mesh = TriMesh(filename)

        self.density = np.zeros((self.mesh.n_points))
        self.vectors = np.random.random((self.mesh.n_points,2))

        self.Interpolator = Interpolator(self.mesh)

    def apply_boundary_condition(self):
        self.vectors[self.mesh.left_id,0] = 0
        self.vectors[self.mesh.right_id,0] = 0
        self.vectors[self.mesh.top_id,1] = 0
        self.vectors[self.mesh.bottom_id,1] = 0

    def computeExternalForces(self, dt):
        pass

    def computeViscosity(self, dt):
        pass

    def computePressure(self, dt):
        x0 = self.poisson_solver()
        grad = self.gradient(x0)

        self.vectors[:,0] -= grad[:,0]
        self.vectors[:,1] -= grad[:,1]

        self.apply_boundary_condition()

    def computeAdvection(self, density, dt):
        new_pos = self.mesh.points - self.vectors*dt
        new_pos = np.clip(new_pos,0,pi)
        if density:
            self.density = self.Interpolator(self.density, new_pos)
        else:
            self.vectors[:,0] = self.Interpolator(self.vectors[:,0], new_pos)
            self.vectors[:,1] = self.Interpolator(self.vectors[:,1], new_pos)

        self.apply_boundary_condition()

    def computeSource(self, dt):
        self.density[[25,40]] = 1

    def divergent(self, pid):
        div = np.sum(self.mesh.rbf[pid][:,1]*self.vectors[self.mesh.nring[pid],0] 
                   + self.mesh.rbf[pid][:,2]*self.vectors[self.mesh.nring[pid],1])
        self.apply_boundary_condition()
        return div
        

    def gradient(self, lapl):
        grad = np.zeros((self.mesh.n_points,2))
        for pid in range(self.mesh.n_points):        
            grad[pid] = (np.sum(self.mesh.rbf[pid][:,1]*lapl[self.mesh.nring[pid]]),
                        np.sum(self.mesh.rbf[pid][:,2]*lapl[self.mesh.nring[pid]]))
        return grad

    def poisson_solver(self):
        lapl = np.asarray([])
        w = np.zeros(self.mesh.n_points)
        b = np.zeros(self.mesh.n_points)
        for pid in range(self.mesh.n_points): 
            if pid in self.mesh.boundary:
                weights = (self.mesh.rbf[pid][:,1]*self.mesh.normals[pid,0] 
                        + self.mesh.rbf[pid][:,2]*self.mesh.normals[pid,1])
            else:
                weights = self.mesh.rbf[pid][:,0]
                b[pid] = self.divergent(pid)
            
            line = np.zeros(self.mesh.n_points)
            line[self.mesh.nring[pid]]= weights
            # p.show(cpos='xy')
            w = np.vstack([w, line])
        w = np.delete(w, 0,0)
        # w[0] = np.identity(mesh.n_points)[0]
        # b[0] = solution(mesh.points[0][0],mesh.points[0][1])
        # np.savetxt("foo.csv", w, fmt="%.7s",delimiter="\t")

        lapl = np.linalg.solve(w,b)
        return lapl

    def velocityStep(self, dt):
        self.computeExternalForces(dt)

        self.computeViscosity(dt)

        # self.computePressure(dt)

        self.computeAdvection(False, dt)

        self.computePressure(dt)

    def densityStep(self, dt):
        self.computeSource(dt)

        self.computeViscosity(dt)

        self.computeAdvection(True, dt)

    def update_fields(self, dt):
        self.apply_boundary_condition()
        self.velocityStep(dt)

        self.apply_boundary_condition()
        self.densityStep(dt)

    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from math import cos, sin
    from matplotlib import cm

    def rbf_problem(x,y):
        return -2*cos(x)*cos(y)
    def rbf_solution(x,y):
        return cos(x)*cos(y) - 1 
    
    mesh = TriMesh("./assets/mesh16.obj")

    plt.triplot(mesh.points[:,0],mesh.points[:,1])
    plt.show(block=True)
    problem = [rbf_problem(p[0],p[1]) for p in mesh.points]
    sol = [rbf_solution(p[0],p[1]) for p in mesh.points]
    # mesh.arrows.plot(cpos='xy')
    lapl = np.asarray([])
    w = np.zeros(mesh.n_points)
    b = np.zeros(mesh.n_points)
    for pid in range(mesh.n_points): 
        if pid in mesh.boundary:
            weights = (mesh.rbf[pid][:,1]*mesh.normals[pid,0] 
                    + mesh.rbf[pid][:,2]*mesh.normals[pid,1])
        else:
            weights = mesh.rbf[pid][:,0]
            b[pid] = problem[pid]
        
        line = np.zeros(mesh.n_points)
        line[mesh.nring[pid]]= weights
        # p.show(cpos='xy')
        w = np.vstack([w, line])
    w = np.delete(w, 0,0)
    w[0] = np.identity(mesh.n_points)[0]
    b[0] = rbf_solution(mesh.points[0][0],mesh.points[0][1])
    # np.savetxt("foo.csv", w, fmt="%.7s",delimiter="\t")

    lapl = np.linalg.solve(w,b)
    error = abs(lapl-sol)
    fig = plt.figure(num='Poisson',figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax1.set_title('Nossa solução')
    ax2.set_title('Solução Exata')
    ax3.set_title('Erro')
    # Plot the surface.
    surf = ax1.plot_trisurf(mesh.points[:,0], mesh.points[:,1], lapl, cmap=cm.coolwarm                        )
    surf = ax2.plot_trisurf(mesh.points[:,0], mesh.points[:,1], sol, cmap=cm.coolwarm                       )
    surf3 = ax3.plot_trisurf(mesh.points[:,0], mesh.points[:,1], error, cmap=cm.coolwarm                        )
    fig.colorbar(surf3, ax=ax3, fraction=0.1, pad=0.2)
    fig.suptitle(f'Norma infito do erro = {max(error):1e}', fontsize=20)
    print(max(error))
    plt.show(block=True)


    def div_problem(x,y):
        return -2*cos(x)*cos(y),-2*cos(x)*cos(y)

    def div_solution(x,y):
        return 2*sin(x+y)
    
    vectors=np.asarray([div_problem(p[0],p[1]) for p in mesh.points])
    div_sol = [div_solution(p[0],p[1]) for p in mesh.points]
    div = np.asarray([np.sum(mesh.rbf[pid][:,1]*vectors[mesh.nring[pid],0]+mesh.rbf[pid][:,2]*vectors[mesh.nring[pid],1]) for pid in range(mesh.n_points)])
    
    error = abs(div-div_sol)
    fig = plt.figure(num='divergence', figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax1.set_title('Nossa solução')
    ax2.set_title('Solução Exata')
    ax3.set_title('Erro')
    fig.suptitle(f'Norma infito do erro = {max(error):1e}',fontsize=20)
    # Plot the surface.
    surf = ax1.plot_trisurf(mesh.points[:,0], mesh.points[:,1], lapl, cmap=cm.coolwarm                        )
    surf = ax2.plot_trisurf(mesh.points[:,0], mesh.points[:,1], sol, cmap=cm.coolwarm                       )
    surf3 = ax3.plot_trisurf(mesh.points[:,0], mesh.points[:,1], error, cmap=cm.coolwarm                        )
    fig.colorbar(surf3, ax=ax3, fraction=0.1, pad=0.2)
    print(max(error))
    plt.show(block=True)