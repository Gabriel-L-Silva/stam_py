import numpy as np
from shapely.geometry import Polygon
from typing import Tuple, List, Optional
from shapely.geometry import LineString, Point, shape, MultiPolygon
import trimesh
from numba import njit, prange
from numba.np.extensions import cross2d
try:
    from modules.generate_mesh import get_geojson, polygon_triangulation
except:
    from generate_mesh import get_geojson, polygon_triangulation
try:
    from modules.tri_mesh import TriMesh
except:
    from tri_mesh import TriMesh

class Ray:
    def __init__(self, ray_origin, ray_direction) -> None:
        self.origin = ray_origin
        self.direction = ray_direction
    
    def __iter__(self):
        yield self.origin
        yield self.direction

@njit
def ray_line_intersection(rayOrigin, rayDirection, edges: np.ndarray):
    point1 = edges[:, 0]
    point2 = edges[:, 1]
    
    v1 = rayOrigin - point1
    v2 = point2 - point1
    v3 = np.array([-rayDirection[1], rayDirection[0]])
    t1 = cross2d(v2, v1) / np.dot(v2, v3)
    t2 = np.dot(v1, v3) / np.dot(v2, v3)
    mask=np.logical_and(np.logical_and(t1 >= 0.0, t1 <= 1.0), np.logical_and(t2 >= 0.0, t2 <= 1.0))
    if len(np.where(mask)[0])==0:
        return np.array([np.nan, np.nan])
    return rayOrigin + min(t1[mask]) * rayDirection

@njit(parallel=True)
def ray_intersects_polygon(vertices:np.ndarray, ray_origins:np.ndarray, ray_directions:np.ndarray, edges:np.ndarray) -> np.ndarray:
    """
    Given a ray specified as a tuple of two points (the starting point and the direction of the ray, each point is a tuple
    of x and y coordinates), and a simple closed polygon specified as a list of points (each point is a tuple of x and y
    coordinates), this function returns True if the ray intersects the polygon and False otherwise.
    """
    intersections = np.zeros((len(ray_origins),2))
    edges = vertices[edges.flatten()].reshape(len(edges),2,3)[:,:,:2]
    # Check if the ray intersects any edge of the polygon
    for r in prange(len(ray_origins)):
        intersection = ray_line_intersection(ray_origins[r], ray_directions[r], edges)
        intersections[r] = intersection
    return intersections

if __name__ == '__main__':
    def angle_between_vectors_in_space(v1, v2):
        angle = np.arctan2(v1[:,1], v1[:,0]) - np.arctan2(v2[:,1],  v2[:,0])
        return angle
    
    scene = trimesh.Scene()
    n_rays = 100
    ray_origins = np.random.uniform(-1,1,size=(n_rays,3))*[100,100,0]
    # ray_origins =np.array([0.0, 0.0, 0.0]).repeat(n_rays).reshape(n_rays, 3)
    # ray_origins = np.array([[58.26558144,  24.24415392,  0.        ]])
    ray_directions = np.random.uniform(-1,1,size=(n_rays,3))*[100,100,0]
    # ray_directions = np.array([100.0, 100.0, 0.0]).repeat(n_rays).reshape(n_rays, 3)
    # ray_directions = np.array([[-200.48438449, -100.41023401,  -0.        ]])
    rays = np.array([list(Ray(o,d)) for o,d in zip(ray_origins, ray_directions)])
    
    current_mesh: int = 0
    geojson, mesh_list = get_geojson()
    mesh_list = list(mesh_list)
    poly: Polygon = shape(geojson[current_mesh]['geometry'])
    if type(poly) == MultiPolygon:
        poly = list(poly)[0]
    mesh, mesh.boundary, scale = polygon_triangulation(poly, 720, 720)
    t_mesh = TriMesh(mesh)
    axis_mesh = trimesh.creation.axis(axis_length=50)
    radius = 0.1
    heights = np.linalg.norm(ray_directions, axis=1)
    vector_angles = angle_between_vectors_in_space(ray_directions, np.array([[0,1,0]]).repeat(n_rays,axis=0))
    transforms = [trimesh.transformations.compose_matrix(translate=t,angles=[-np.pi/2, 0, a]) for t,a in zip(ray_origins, vector_angles)]
    caps = [trimesh.primitives.Capsule(radius=radius, height=height, transform=transform, colors=[0,255,0]) for height, transform in zip(heights, transforms)]
    scene.add_geometry(axis_mesh)
    scene.add_geometry(t_mesh.mesh)
    scene.add_geometry(caps)
    
    intersections = ray_intersects_polygon(mesh.vertices, rays, np.array(t_mesh.sorted_edges))
    
    intersections = intersections[np.isfinite(intersections).all(axis=1)]
    if intersections.size != 0:
        intersections = np.stack([intersections[:,0],intersections[:,1],np.zeros((len(intersections)))], axis=1).reshape(-1,3)
        points = trimesh.PointCloud(intersections,colors=[255,0,0])
        scene.add_geometry(points)
    scene.show(smooth=False, flags={'wireframe':True})

# [[-95.26336044 -52.64900173]]