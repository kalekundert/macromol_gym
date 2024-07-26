import macromol_gym_unsupervised.random as _mmgu
import numpy as np

from scipy.stats import ks_1samp

def test_sample_uniform_unit_vector():
    # The following references give the distribution for the distance between 
    # two random points on a unit sphere of any dimension:
    #
    # https://math.stackexchange.com/questions/4654438/distribution-of-distances-between-random-points-on-spheres
    # https://johncarlosbaez.wordpress.com/2018/07/10/random-points-on-a-sphere-part-1/
    #
    # For the 3D case, the PDF is remarkably simple:
    #
    #   p(d) = d/2
    #
    # Here, we use the 1-sample KS test to check that our sampled distribution 
    # is consistent with this expected theoretical distribution.
    
    n = 1000
    rng = np.random.default_rng(0)

    d = np.zeros(n)
    x = np.array([1, 0, 0])  # arbitrary reference point

    for i in range(n):
        y = _mmgu.sample_uniform_unit_vector(rng)
        d[i] = np.linalg.norm(y - x)

    cdf = lambda d: d**2 / 4
    ks = ks_1samp(d, cdf)

    # This test should fail for 5% of random seeds, but 0 is one that passes.
    assert ks.pvalue > 0.05

def test_sample_coord_from_cube():
    # Don't test that the sampling is actually uniform; I think this would be 
    # hard to do, and the implementation is simple enough that there's probably 
    # not a mistake.  Instead, just check that the points are all within the 
    # cube.

    rng = np.random.default_rng(0)

    n = 1000
    x = np.zeros((n,3))
    center = np.array([0, 2, 4])
    size = 4

    for i in range(n):
        x[i] = _mmgu.sample_coord_from_cube(rng, center, size)

    # Check that all the points are in-bounds:
    assert np.all(x[:,0] >= -2)
    assert np.all(x[:,0] <=  2)

    assert np.all(x[:,1] >  0)
    assert np.all(x[:,1] <  4)

    assert np.all(x[:,2] >  2)
    assert np.all(x[:,2] <  6)

    # Check that the sampling is uniform in each dimension:
    def cdf(a, b):
        return lambda x: (x - a) / (b - a)

    ks_x = ks_1samp(x[:,0], cdf(-2, 2))
    ks_y = ks_1samp(x[:,1], cdf( 0, 4))
    ks_z = ks_1samp(x[:,2], cdf( 2, 6))

    # Each of these tests should fail for 5% of random seeds, but 0 is one that 
    # passes.
    assert ks_x.pvalue > 0.05
    assert ks_y.pvalue > 0.05
    assert ks_z.pvalue > 0.05

