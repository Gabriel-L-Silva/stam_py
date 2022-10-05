from math import pi
import numpy as np
import trimesh

def main():
    size = 128
    x= np.linspace(0,pi,size+1)
    y= np.linspace(0,pi,size+1)
    xx, yy = np.meshgrid(x,y)
    
    points = np.array([xx.flatten(),yy.flatten(),np.zeros(len(yy.flatten()))])
    indices = np.zeros(size**2*6,dtype=int)
    index=0
    for x in range(size):
      for z in range(size):
        offset = x * (size+1) + z
        indices[index] = (offset + 0)
        indices[index + 1] = (offset + 1)
        indices[index + 2] = (offset + size+1)
        indices[index + 3] = (offset + 1)
        indices[index + 4] = (offset + size+1 + 1)
        indices[index + 5] = (offset + size+1)
        index += 6
    tmesh = trimesh.Trimesh(points.T, indices.reshape(-1,3))
    tmesh.export(f'assets/regular_tri_grid{size}.obj')
    tmesh.show(smooth=False, flags={'wireframe':True})

if __name__ == '__main__':
    main()