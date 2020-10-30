import numpy as np
import pandas as pd
from pytest import approx

from mrktmix.dataprep import mmm_transform as dp


def test_apply_apl_lagisint():
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


def test_apply_apl_series():
    # apply apl on pandas series
    dates = (pd.date_range(start='20180601', end='20180630', freq='W-FRI')
             .strftime('%Y-%m-%d'))
    df = pd.Series(np.array([1., 2., 3., 4., 5.]), index=dates, name="two")
    input_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    output_single = pd.Series(np.array(input_list), index=dates, name=('two', 0, 1, 0))
    pd.testing.assert_series_equal(output_single, dp.apply_apl_series(df, 0, 1, 0))
    # apply apl on pandas series
    dates = (pd.date_range(start='20180601', end='20180630', freq='W-FRI')
             .strftime('%Y-%m-%d'))
    df = pd.Series(np.array([1., 2., 3., 4., 5.]), index=dates, name="two")
    input_list = [0.0, 1.0, 1.8991444823309347, 2.753418658226993, 3.5561024564142]
    output_single = pd.Series(np.array(input_list), index=dates, name=('two', .50, .70, 1))
    pd.testing.assert_series_equal(output_single, dp.apply_apl_series(df, 0.5, .7, 1))
