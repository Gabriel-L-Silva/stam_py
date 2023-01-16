import numpy as np
from shapely.geometry import Polygon
from typing import Tuple, List, Optional
from shapely.geometry import LineString, Point, shape, MultiPolygon
import trimesh

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

def line_intersect(ray: Ray, edge):
    ray = LineString([tuple(ray.origin[:2]), tuple(ray.origin+ ray.direction)[:2]])
    line = LineString([edge[0], edge[1]])

    intersection = ray.intersection(line)

    return not intersection.is_empty, intersection

def ray_intersects_polygon(mesh, ray, edges) -> bool:
    """
    Given a ray specified as a tuple of two points (the starting point and the direction of the ray, each point is a tuple
    of x and y coordinates), and a simple closed polygon specified as a list of points (each point is a tuple of x and y
    coordinates), this function returns True if the ray intersects the polygon and False otherwise.
    """
    intersections = []
    # Check if the ray intersects any edge of the polygon
    for edge in edges:
        intersect, intersection = line_intersect(ray, mesh.vertices[edge])
        if intersect:
            intersections.append([edge,intersection])
    return intersections

def is_point_on_infinite_line(point, line) -> bool:
    """
    Given a point specified as a tuple of x and y coordinates, and a line specified as a tuple of two points (each point is
    a tuple of x and y coordinates), this function returns True if the point lies on the infinite line defined by the line
    and False otherwise.
    """
    # Check if the point lies on the infinite line defined by the line
    if (line[1][1] - line[0][1]) * (point[0] - line[0][0]) == (line[1][0] - line[0][0]) * (point[1] - line[0][1]):
        return True
    else:
        return False

def is_point_on_segment(point, segment) -> bool:
    """
    Given a point specified as a tuple of x and y coordinates, and a line segment specified as a tuple of two points (each
    point is a tuple of x and y coordinates), this function returns True if the point lies on the segment and False otherwise.
    """
    # Check if the point lies on the infinite line defined by the segment
    if not is_point_on_infinite_line(point, segment):
        return False
    
    # Check if the point lies within the segment
    x_coordinates = [segment[0][0], segment[1][0]]
    y_coordinates = [segment[0][1], segment[1][1]]
    if min(x_coordinates) <= point[0] <= max(x_coordinates) and min(y_coordinates) <= point[1] <= max(y_coordinates):
        return True
    else:
        return False

def distance(point1, point2):
    """
    Given two points specified as tuples of x and y coordinates, this function returns the distance between the points.
    """
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def ray_intersection_point(mesh, ray, edges, poly):
    """
    Given a ray specified as a tuple of two points (the starting point and the direction of the ray, each point is a tuple
    of x and y coordinates), and a simple closed shapely polygon, this function returns the intersection point of the ray 
    with the polygon if it exists, or None if the ray does not intersect the polygon.
    """
    line = LineString([ray.origin[:2], ray.direction[:2]-ray.origin[:2]])
    from shapely.geometry.polygon import LinearRing
    lring = LinearRing(list(poly.exterior.coords))
    # Determine whether the ray and the polygon intersect
    edge_intersections = ray_intersects_polygon(mesh, ray, edges)
    if not edge_intersections:
        return None
    
    # Loop through each edge of the polygon
    candidate_intersection_points = []
    ray_line = [tuple(ray.origin[:2]), tuple(ray.direction[:2])-ray.origin[:2]]
    for edge, intersection in edge_intersections:
        edge = mesh.vertices[edge]
        candidate_intersection_points.append(np.copy(intersection.coords[0][:-1]))
    
    # Choose the candidate intersection point that is closest to the starting point of the ray, and return it as the intersection point of the ray with the polygon
    if candidate_intersection_points:
        return min(candidate_intersection_points, key=lambda p: distance(ray.origin, p))
    else:
        return None

def angle_between_vectors_in_space(v1, v2):
    angle = np.arctan2(v1[:,1], v1[:,0]) - np.arctan2(v2[:,1],  v2[:,0])
    return angle

if __name__ == '__main__':
    scene = trimesh.Scene()
    n_rays = 100
    ray_origins = np.random.uniform(-1,1,size=(n_rays,3))*[100,100,0]
    # ray_origins =np.array([0.0, 0.0, 0.0]).repeat(n_rays).reshape(n_rays, 3)
    # ray_origins = np.array([[58.26558144,  24.24415392,  0.        ]])
    ray_directions = np.random.uniform(-1,1,size=(n_rays,3))*[100,100,0]
    # ray_directions = np.array([100.0, 100.0, 0.0]).repeat(n_rays).reshape(n_rays, 3)
    # ray_directions = np.array([[-200.48438449, -10.41023401,  -0.        ]])
    rays = [Ray(o,d) for o,d in zip(ray_origins, ray_directions)]
    
    current_mesh: int = 0
    geojson, mesh_list = get_geojson()
    mesh_list = list(mesh_list)
    poly: Polygon = shape(geojson[current_mesh]['geometry'])
    if type(poly) == MultiPolygon:
        poly = list(poly)[0]
    mesh = polygon_triangulation(poly, 720, 720)
    mesh.boundary = poly
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
    
    intersections = np.array([ray_intersection_point(mesh, ray, t_mesh.sorted_edges, poly) for ray in rays if ray_intersection_point(mesh, ray, t_mesh.sorted_edges,poly) is not None])
    
    if intersections.size != 0:
        intersections = np.stack([intersections[:,0],intersections[:,1],np.zeros((len(intersections)))], axis=1).reshape(-1,3)
        points = trimesh.PointCloud(intersections,colors=[255,0,0])
        scene.add_geometry(points)
    scene.show(smooth=False, flags={'wireframe':True})

