import macromol_gym as mmg
import numpy as np

from pytest import approx

def test_tetrahedron_faces():
    v = mmg.tetrahedron_faces()

    assert len(v) == 4
    assert np.linalg.norm(v, axis=1) == approx(1)

def test_cube_faces():
    v = mmg.cube_faces()

    assert len(v) == 6
    assert np.linalg.norm(v, axis=1) == approx(1)

def test_octahedron_faces():
    v = mmg.octahedron_faces()

    assert len(v) == 8
    assert np.linalg.norm(v, axis=1) == approx(1)

def test_dodecahedron_faces():
    v = mmg.dodecahedron_faces()

    assert len(v) == 12
    assert np.linalg.norm(v, axis=1) == approx(1)

def test_icosahedron_faces():
    v = mmg.icosahedron_faces()

    assert len(v) == 20
    assert np.linalg.norm(v, axis=1) == approx(1)

