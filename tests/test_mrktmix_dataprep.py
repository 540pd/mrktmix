from pytest import approx

from mrktmix.dataprep import mmm_transform as dp

# apl when lag is integer


def test_apply_apl_lagIsint():
    # same output
    assert all(dp.apply_apl_([100, 50, 10, 0, 0, 0], 0, 1, 0) == [100., 50., 10., 0., 0., 0])
    # adstock
    assert all(dp.apply_apl_([100, 50, 10, 0, 0, 0], .5, 1, 0) == [100., 100., 60., 30., 15., 7.5])
    # power
    assert dp.apply_apl_([100, 50, 10, 0], 0, .9, 0) == approx([63.09573445, 33.81216689, 7.94328235, 0.])
    # lag
    assert all(dp.apply_apl_([100, 50, 10, 0, 0, 0], 0, 1, 4) == [0., 0., 0., 0., 100., 50.])
    # combined
    assert dp.apply_apl_([100, 50, 10, 0, 0, 0], .5, .4, 2) == approx([0., 0., 6.30957344, 6.30957344, 5.1435208, 3.89805984])
