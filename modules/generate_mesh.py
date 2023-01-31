from math import pi
import numpy as np
import trimesh
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, shape
import geojson

def regular_tri_grid(size = 64):
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
    return points, indices

def TS(polygon, W, H):
    x_min, y_min, x_max, y_max = polygon.bounds
    cx = polygon.centroid.x
    cy = polygon.centroid.y
    Sx = W/(x_max-x_min)
    Sy = H/(y_max-y_min)

    scale_factor = Sx if Sx < Sy else Sy

    poly = affine_transform(polygon, [scale_factor, 0, 0, scale_factor, -cx*scale_factor, -cy*scale_factor])
    #move to origin
    return poly, scale_factor

def polygon_triangulation(polygon: Polygon, W, H, max_tri_area):
    poly, scale_factor = TS(polygon, W, H)
    points, indices = trimesh.creation.triangulate_polygon(poly, 'pq30a'+str(max_tri_area if max_tri_area>0.0 else 1), engine='triangle')
    return trimesh.Trimesh(np.stack([points[:,0], points[:,1], np.zeros(len(points))], axis = 1), indices), poly, scale_factor

def get_geojson():
    '''
    in: filename to geojson
    out: json, list of names
    '''
    with open('./assets/geo-countries/archive/countries.geojson') as f:
        data = geojson.load(f)
    return data, np.array([f['properties']['ADMIN'] for f in data['features']])

def main():
    # size = 4
    # points, indices = regular_tri_grid(size)
    # tmesh = trimesh.Trimesh(points.T, indices.reshape(-1,3))
    # # tmesh.export(f'assets/regular_tri_grid{size}.obj')
    # tmesh.show(smooth=False, flags={'wireframe':True})

    
    data, names = get_geojson()
    idx = np.where(names=='Niger')[0][0]
    poly: Polygon = shape(data['features'][idx]['geometry'])
    tmesh= polygon_triangulation(poly)
    tmesh.show(smooth=False, flags={'wireframe':True})

if __name__ == '__main__':
    main()