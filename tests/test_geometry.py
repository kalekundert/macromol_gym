import macromol_training as mmt
import numpy as np

from pytest import approx

def test_tetrahedron_faces():
    v = mmt.tetrahedron_faces()

    assert len(v) == 4
    assert np.linalg.norm(v, axis=1) == approx(1)

def test_cube_faces():
    v = mmt.cube_faces()

    assert len(v) == 6
    assert np.linalg.norm(v, axis=1) == approx(1)

def test_octahedron_faces():
    v = mmt.octahedron_faces()

    assert len(v) == 8
    assert np.linalg.norm(v, axis=1) == approx(1)

def test_dodecahedron_faces():
    v = mmt.dodecahedron_faces()

    assert len(v) == 12
    assert np.linalg.norm(v, axis=1) == approx(1)

def test_icosahedron_faces():
    v = mmt.icosahedron_faces()

    assert len(v) == 20
    assert np.linalg.norm(v, axis=1) == approx(1)

