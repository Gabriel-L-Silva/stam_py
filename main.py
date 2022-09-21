import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import pi
from modules.trisolver import TriSolver
    
def update(frame, solver):
    
    solver.update_field()
    im.set_array(solver.density)
    vc.U = solver.vectors[:,0]
    vc.V = solver.vectors[:,1]
    
def main():
    global im, vc

    timeInterval = 1/60.0

    filename = './assets/mesh8.obj'
    solver = TriSolver(filename, timeInterval)

    fig, ax = plt.subplots()
    ax.set(xlim=(0,pi),ylim=(0,pi))
    #criando a figura e adiocionando os eixos

    # cid = fig.canvas.mpl_connect('button_press_event', on_button_press)
    # cid = fig.canvas.mpl_connect('button_release_event', on_button_release)
    # cid = fig.canvas.mpl_connect('motion_notify_event', on_motion)
    #mpl_connect mapeia para um inteiro ou eu que faço?
    # onde ele muda esse cid ao longo do código
    #im é a imagem que vamos trabalhar em cima, im.set_data() atualiza a im a cada chamada
    im = ax.tripcolor(solver.mesh.points[:,0], solver.mesh.points[:,1], solver.density, shading='gouraud',cmap=plt.cm.gray)
    vc = plt.quiver(solver.mesh.points[:,0],solver.mesh.points[:,1],solver.vectors[:,0],solver.vectors[:,1],color='red')
    #[1:-1, 1:-1]: vai da primeira ate a ultima posiçao do array
    #extent: espaço que a imagem vai ocupar
    #vmin e vmax: range do color map
    animation = FuncAnimation(fig, update, interval=10, frames=800, fargs=(solver,))
    #interval: delay entre os frames em milissegundos, super rápido
    plt.show()

if __name__ == '__main__':
    main()