import numpy as np
import functools
from math import sqrt

PHI = (1 + sqrt(5)) / 2

@functools.cache
def tetrahedron_faces():
    x = 1 / sqrt(2)

    # The vertices of a regular tetrahedron are equivalent to its faces:
    # https://en.wikipedia.org/wiki/Tetrahedron#Coordinates_for_a_regular_tetrahedron

    verts = np.array([
        [ 1,  0,  -x],
        [-1,  0,  -x],
        [ 0,  1,   x],
        [ 0, -1,   x],
    ])
    norm = sqrt(1 + x**2)

    return verts / norm

@functools.cache
def cube_faces():
    # I chose the order of these coordinates in order to match the original 
    # atompaint dataset.  That's why they're a bit "out of order".
    return np.array([
        [ 0, -1,  0],
        [ 0,  1,  0],
        [-1,  0,  0],
        [ 1,  0,  0],
        [ 0,  0,  1],
        [ 0,  0, -1],
    ])

@functools.cache
def octahedron_faces():
    # The vertices of a regular cube are the faces of a regular octahedron:
    # https://en.wikipedia.org/wiki/Cube#Cartesian_coordinates
    verts = np.array([
        [ 1,  1,  1],
        [ 1,  1, -1],
        [ 1, -1,  1],
        [ 1, -1, -1],
        [-1,  1,  1],
        [-1,  1, -1],
        [-1, -1,  1],
        [-1, -1, -1],
    ])
    return verts / sqrt(3)

@functools.cache
def dodecahedron_faces():
    # The vertices of a regular icosahedron are the faces of a regular 
    # dodecahedron:
    # https://en.wikipedia.org/wiki/Regular_icosahedron#Construction
    verts = np.array([
        [   0,    1,  PHI],
        [   0,   -1,  PHI],
        [   0,    1, -PHI],
        [   0,   -1, -PHI],
        [   1,  PHI,    0],
        [  -1,  PHI,    0],
        [   1, -PHI,    0],
        [  -1, -PHI,    0],
        [ PHI,    0,    1],
        [ PHI,    0,   -1],
        [-PHI,    0,    1],
        [-PHI,    0,   -1],
    ])
    norm = sqrt(1 + PHI**2)

    return verts / norm

@functools.cache
def icosahedron_faces():
    # The vertices of a regular dodecahedron are the faces of a regular 
    # icosahedron:
    # https://en.wikipedia.org/wiki/Regular_dodecahedron#Cartesian_coordinates
    verts = np.array([
        [     1,      1,      1],
        [     1,      1,     -1],
        [     1,     -1,      1],
        [     1,     -1,     -1],
        [    -1,      1,      1],
        [    -1,      1,     -1],
        [    -1,     -1,      1],
        [    -1,     -1,     -1],

        [     0,    PHI,  1/PHI],
        [     0,    PHI, -1/PHI],
        [     0,   -PHI,  1/PHI],
        [     0,   -PHI, -1/PHI],

        [ 1/PHI,      0,    PHI],
        [-1/PHI,      0,    PHI],
        [ 1/PHI,      0,   -PHI],
        [-1/PHI,      0,   -PHI],

        [   PHI,  1/PHI,      0],
        [   PHI, -1/PHI,      0],
        [  -PHI,  1/PHI,      0],
        [  -PHI, -1/PHI,      0],
    ])
    return verts / sqrt(3)


def _calc_edge_indices(verts):
    from scipy.spatial import distance_matrix
    from more_itertools import unique_everseen as unique

    d = distance_matrix(verts, verts)
    edge_dist = d[d > 0].min()

    # It's dangerous to compare floating point numbers like this (any small 
    # difference in rounding will result in two numbers that aren't equal), but 
    # it seems to work for regular polyhedra, perhaps because the numbers 
    # comprising each vertex differ only by sign, not value.
    i, j = np.where(np.triu(d) == edge_dist)

    return list(unique(zip(i, j)))

def _calc_face_indices(verts):
    import networkx as nx

    edges = _calc_edge_indices(verts)
    graph = nx.Graph(edges)

    return sorted(nx.simple_cycles(graph, 3))


if __name__ == '__main__':
    from pprint import pprint

    # I originally wrote the `calc_*_indices()` functions when I thought I had 
    # to work out face coordinates from vertex coordinates.  My intention was 
    # to use these functions to generate a list of vertex indices that I could 
    # then hard-code into face coordinate functions.  I eventually realized 
    # that each regular polyhedra is dual to another regular polyhedra, so the 
    # face coordinates of one are just the vertex coordinates of another.  This 
    # is a much better, simpler approach.  However, I kept the code for 
    # calculating face coordinates, because it wasn't trivial to write and it 
    # might be useful later.

    v = dodecahedron_faces()
    print("Icosahedron edge indices:")
    pprint(_calc_edge_indices(v))
    print()
    print("Icosahedron face indices:")
    pprint(_calc_face_indices(v))





