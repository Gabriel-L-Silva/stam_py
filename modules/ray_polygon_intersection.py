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
    ray = LineString([tuple(ray.origin[:2]), (ray.origin[0] + ray.direction[0], ray.origin[1] + ray.direction[1])])
    line = LineString([edge[0], edge[1]])

    intersection = ray.intersection(line)

    return not intersection.is_empty, intersection

def ray_intersects_polygon(mesh, ray, edges) -> bool:
    """
    Given a ray specified as a tuple of two points (the starting point and the direction of the ray, each point is a tuple
    of x and y coordinates), and a simple closed polygon specified as a list of points (each point is a tuple of x and y
    coordinates), this function returns True if the ray intersects the polygon and False otherwise.
    """
    # Check if the ray intersects any edge of the polygon
    for edge in edges:
        if line_intersect(ray, mesh.vertices[edge]):
            return True

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

def ray_intersection_point(mesh, ray, edges):
    """
    Given a ray specified as a tuple of two points (the starting point and the direction of the ray, each point is a tuple
    of x and y coordinates), and a simple closed shapely polygon, this function returns the intersection point of the ray 
    with the polygon if it exists, or None if the ray does not intersect the polygon.
    """
    # Determine whether the ray and the polygon intersect
    if not ray_intersects_polygon(mesh, ray, edges):
        return None
    
    # Loop through each edge of the polygon
    candidate_intersection_points = []
    ray_line = [tuple(ray.origin[:2]), (ray.origin[0] + ray.direction[0], ray.origin[1] + ray.direction[1])]
    for edge in edges:
        edge = mesh.vertices[edge]
        # Calculate the intersection point of the ray with the current edge
        intersect, intersection_point = line_intersect(ray, edge)
        
        # If the intersection point lies on the ray and within the edge, store it as a candidate intersection point
        check = intersect and is_point_on_segment(intersection_point.coords[0][:-1], ray_line) and is_point_on_segment(intersection_point.coords[0][:-1], edge)
        if check:
            candidate_intersection_points.append(np.copy(intersection_point.coords[0][:-1]))
    
    # Choose the candidate intersection point that is closest to the starting point of the ray, and return it as the intersection point of the ray with the polygon
    if candidate_intersection_points:
        return min(candidate_intersection_points, key=lambda p: distance(ray.origin, p))
    else:
        return None

if __name__ == '__main__':
    scene = trimesh.Scene()

    ray_origin = np.array([1.0, 1.0, 0.0])
    ray_direction = np.array([100.0, 100.0, 0.0])
    ray = Ray(ray_origin, ray_direction)
    
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
    entity= trimesh.path.entities.Line([ray.origin.tolist(), (ray.origin+ray.direction).tolist()])
    path = trimesh.path.Path2D([entity], entity.end_points.tolist())
    path.show()
    scene.add_geometry(axis_mesh)
    scene.add_geometry(t_mesh.mesh)
    scene.add_geometry(path)
    scene.show(smooth=False, flags={'wireframe':True})
    # t_mesh.mesh.show(smooth=False, flags={'wireframe':True})

    intersection = ray_intersection_point(mesh, ray, t_mesh.sorted_edges)

    print(intersection)

