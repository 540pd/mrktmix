import numpy as np
import pandas as pd
from pytest import approx

from mrktmix.dataprep import transform as dp


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


def test_segregate_variable():
    # simple case
    aggregated_data = pd.Series([51, 72, 51, 19, 47, 39, 33], pd.date_range("2020-01-01", "2020-01-07"), name="ABC")
    segregated_data = pd.DataFrame([[35, 96, 16],
                                    [23, 97, 79],
                                    [77, 64, 79],
                                    [16, 29, 60],
                                    [66, 52, 87],
                                    [51, 98, 84],
                                    [69, 73, 19]], pd.date_range("2020-01-01", "2020-01-07"), columns=["M", "N", "O"])
    output_data = pd.DataFrame([[12.14285714, 33.30612245, 5.55102041],
                                [8.32160804, 35.09547739, 28.58291457],
                                [17.85, 14.83636364, 18.31363636],
                                [2.8952381, 5.24761905, 10.85714286],
                                [15.13170732, 11.92195122, 19.94634146],
                                [8.53648069, 16.40343348, 14.06008584],
                                [14.14285714, 14.96273292, 3.89440994]], pd.date_range("2020-01-01", "2020-01-07"), columns=["M", "N", "O"])
    pd.testing.assert_frame_equal(dp.segregate_variable(aggregated_data, segregated_data, match_sum=False),
                                  output_data)
    pd.testing.assert_frame_equal(dp.segregate_variable(aggregated_data, segregated_data, match_sum=True),
                                  output_data)

    # with 0
    aggregated_data = pd.Series([51, 0, 51, 19, 47, 39, 20], pd.date_range("2020-01-01", "2020-01-07"), name="ABC")
    segregated_data = pd.DataFrame([[0, 96, 16],
                                    [23, 97, 79],
                                    [77, 64, 79],
                                    [16, 29, 30],
                                    [66, 0, 87],
                                    [51, 98, 84],
                                    [0, 73, 19]], pd.date_range("2020-01-01", "2020-01-07"), columns=["M", "N", "O"])
    output_data = pd.DataFrame([[0., 43.71428571, 7.28571429],
                                [0., 0., 0.],
                                [17.85, 14.83636364, 18.31363636],
                                [4.05333333, 7.34666667, 7.6],
                                [20.2745098, 0., 26.7254902],
                                [8.53648069, 16.40343348, 14.06008584],
                                [0., 15.86956522, 4.13043478]], pd.date_range("2020-01-01", "2020-01-07"), columns=["M", "N", "O"])
    pd.testing.assert_frame_equal(dp.segregate_variable(aggregated_data, segregated_data, match_sum=False),
                                  output_data)
    pd.testing.assert_frame_equal(dp.segregate_variable(aggregated_data, segregated_data, match_sum=True),
                                  output_data)

    # with mismatch
    aggregated_data = pd.Series([51, 20, 51, 19, 47, 39, 20], pd.date_range("2020-01-01", "2020-01-07"), name="ABC")
    segregated_data = pd.DataFrame([[0, 0, 0],
                                    [23, 97, 79],
                                    [77, 64, 79],
                                    [16, 29, 30],
                                    [66, 10, 87],
                                    [51, 98, 84],
                                    [0, 73, 0]], pd.date_range("2020-01-01", "2020-01-07"), columns=["M", "N", "O"])
    output_data = pd.DataFrame([[np.nan, np.nan, np.nan],
                                [2.31155779, 9.74874372, 7.93969849],
                                [17.85, 14.83636364, 18.31363636],
                                [4.05333333, 7.34666667, 7.6],
                                [19.03067485, 2.88343558, 25.08588957],
                                [8.53648069, 16.40343348, 14.06008584],
                                [0., 20., 0.]], pd.date_range("2020-01-01", "2020-01-07"), columns=["M", "N", "O"])
    pd.testing.assert_frame_equal(dp.segregate_variable(aggregated_data, segregated_data, match_sum=False),
                                  output_data)
    output_data = pd.DataFrame([[np.nan, np.nan, np.nan],
                                [2.91303456, 12.28540662, 10.00564045],
                                [22.49464286, 18.69684601, 23.07891929],
                                [5.10802721, 9.25829932, 9.57755102],
                                [23.98253412, 3.63371729, 31.61334043],
                                [10.7577078, 20.67167382, 17.71857756],
                                [0., 25.20408163, 0.]], pd.date_range("2020-01-01", "2020-01-07"), columns=["M", "N", "O"])
    pd.testing.assert_frame_equal(dp.segregate_variable(aggregated_data, segregated_data, match_sum=True),
                                  output_data)


def test_segregate_panel():
    # when segregated and aggregate data matches i.e. both data are aligned
    segregate_data = pd.DataFrame({'A': {('METRO',
                                          '1/1/2018'): 773,
                                         ('METRO',
                                          '2/1/2018'): 137,
                                         ('METRO',
                                          '3/1/2018'): 508,
                                         ('REGIONAL',
                                          '1/1/2018'): 500,
                                         ('REGIONAL',
                                          '2/1/2018'): 100,
                                         ('REGIONAL',
                                          '3/1/2018'): 400},
                                   'B': {('METRO',
                                          '1/1/2018'): 848,
                                         ('METRO',
                                          '2/1/2018'): 326,
                                         ('METRO',
                                          '3/1/2018'): 969,
                                         ('REGIONAL',
                                          '1/1/2018'): 137,
                                         ('REGIONAL',
                                          '2/1/2018'): 508,
                                         ('REGIONAL',
                                          '3/1/2018'): 562},
                                   'C': {('METRO',
                                          '1/1/2018'): 292,
                                         ('METRO',
                                          '2/1/2018'): 867,
                                         ('METRO',
                                          '3/1/2018'): 323,
                                         ('REGIONAL',
                                          '1/1/2018'): 326,
                                         ('REGIONAL',
                                          '2/1/2018'): 969,
                                         ('REGIONAL',
                                          '3/1/2018'): 730}})
    aggregate_data = segregate_data.sum(level=-1)
    output_index = pd.MultiIndex.from_tuples([('METRO', '1/1/2018'),
                                              ('METRO', '2/1/2018'),
                                              ('METRO', '3/1/2018'),
                                              ('REGIONAL', '1/1/2018'),
                                              ('REGIONAL', '2/1/2018'),
                                              ('REGIONAL', '3/1/2018')],)
    output_data = pd.DataFrame([[773., 848., 292.],
                                [137., 326., 867.],
                                [508., 969., 323.],
                                [500., 137., 326.],
                                [100., 508., 969.],
                                [400., 562., 730.]], output_index, columns=["A", "B", "C"])
    pd.testing.assert_frame_equal(dp.segregate_panel(aggregate_data, segregate_data, match_sum=True),
                                  output_data)
    pd.testing.assert_frame_equal(dp.segregate_panel(aggregate_data, segregate_data, match_sum=False),
                                  output_data)

    # when segregated data have few 0
    segregated_data = pd.DataFrame({'A': {('METRO',
                                           '1/1/2018'): 0,
                                          ('METRO',
                                           '2/1/2018'): 137,
                                          ('METRO',
                                           '3/1/2018'): 508,
                                          ('REGIONAL',
                                           '1/1/2018'): 500,
                                          ('REGIONAL',
                                           '2/1/2018'): 100,
                                          ('REGIONAL',
                                           '3/1/2018'): 400},
                                    'B': {('METRO',
                                           '1/1/2018'): 848,
                                          ('METRO',
                                           '2/1/2018'): 0,
                                          ('METRO',
                                           '3/1/2018'): 969,
                                          ('REGIONAL',
                                           '1/1/2018'): 137,
                                          ('REGIONAL',
                                           '2/1/2018'): 508,
                                          ('REGIONAL',
                                           '3/1/2018'): 562},
                                    'C': {('METRO',
                                           '1/1/2018'): 292,
                                          ('METRO',
                                           '2/1/2018'): 867,
                                          ('METRO',
                                           '3/1/2018'): 0,
                                          ('REGIONAL',
                                           '1/1/2018'): 326,
                                          ('REGIONAL',
                                           '2/1/2018'): 969,
                                          ('REGIONAL',
                                           '3/1/2018'): 0}})
    aggregated_data = segregated_data.sum(level=-1)
    output_index = pd.MultiIndex.from_tuples([('METRO', '1/1/2018'),
                                              ('METRO', '2/1/2018'),
                                              ('METRO', '3/1/2018'),
                                              ('REGIONAL', '1/1/2018'),
                                              ('REGIONAL', '2/1/2018'),
                                              ('REGIONAL', '3/1/2018')],)
    output_data = pd.DataFrame([[0., 848., 292.],
                                [137., 0., 867.],
                                [508., 969., np.nan],
                                [500., 137., 326.],
                                [100., 508., 969.],
                                [400., 562., np.nan]], output_index, columns=["A", "B", "C"])
    pd.testing.assert_frame_equal(dp.segregate_panel(aggregated_data, segregated_data, match_sum=True),
                                  output_data)
    pd.testing.assert_frame_equal(dp.segregate_panel(aggregated_data, segregated_data, match_sum=False),
                                  output_data)

    # when segregated and aggregate are not aligned
    segregate_data_ori = pd.DataFrame({'A': {('METRO',
                                              '1/1/2018'): 773,
                                             ('METRO',
                                              '2/1/2018'): 137,
                                             ('METRO',
                                              '3/1/2018'): 508,
                                             ('REGIONAL',
                                              '1/1/2018'): 500,
                                             ('REGIONAL',
                                              '2/1/2018'): 100,
                                             ('REGIONAL',
                                              '3/1/2018'): 400},
                                       'B': {('METRO',
                                              '1/1/2018'): 848,
                                             ('METRO',
                                              '2/1/2018'): 326,
                                             ('METRO',
                                              '3/1/2018'): 969,
                                             ('REGIONAL',
                                              '1/1/2018'): 137,
                                             ('REGIONAL',
                                              '2/1/2018'): 508,
                                             ('REGIONAL',
                                              '3/1/2018'): 562},
                                       'C': {('METRO',
                                              '1/1/2018'): 292,
                                             ('METRO',
                                              '2/1/2018'): 867,
                                             ('METRO',
                                              '3/1/2018'): 323,
                                             ('REGIONAL',
                                              '1/1/2018'): 326,
                                             ('REGIONAL',
                                              '2/1/2018'): 969,
                                             ('REGIONAL',
                                              '3/1/2018'): 730}})
    aggregated_data = segregate_data_ori.sum(level=-1)
    segregated_data = pd.DataFrame({'A': {('METRO',
                                           '1/1/2018'): 0,
                                          ('METRO',
                                           '2/1/2018'): 137,
                                          ('METRO',
                                           '3/1/2018'): 508,
                                          ('REGIONAL',
                                           '1/1/2018'): 500,
                                          ('REGIONAL',
                                           '2/1/2018'): 100,
                                          ('REGIONAL',
                                           '3/1/2018'): 400},
                                    'B': {('METRO',
                                           '1/1/2018'): 848,
                                          ('METRO',
                                           '2/1/2018'): 0,
                                          ('METRO',
                                           '3/1/2018'): 969,
                                          ('REGIONAL',
                                           '1/1/2018'): 137,
                                          ('REGIONAL',
                                           '2/1/2018'): 508,
                                          ('REGIONAL',
                                           '3/1/2018'): 562},
                                    'C': {('METRO',
                                           '1/1/2018'): 292,
                                          ('METRO',
                                           '2/1/2018'): 867,
                                          ('METRO',
                                           '3/1/2018'): 0,
                                          ('REGIONAL',
                                           '1/1/2018'): 326,
                                          ('REGIONAL',
                                           '2/1/2018'): 969,
                                          ('REGIONAL',
                                           '3/1/2018'): 0}})
    output_index = pd.MultiIndex.from_tuples([('METRO', '1/1/2018'),
                                              ('METRO', '2/1/2018'),
                                              ('METRO', '3/1/2018'),
                                              ('REGIONAL', '1/1/2018'),
                                              ('REGIONAL', '2/1/2018'),
                                              ('REGIONAL', '3/1/2018')],)
    output_data = pd.DataFrame([[0., 848., 417.29584352],
                                [137., 0., 1239.02567237],
                                [508., 969., np.nan],
                                [1273., 137., 465.88508557],
                                [100., 834., 1384.79339853],
                                [400., 562., np.nan]], output_index, columns=["A", "B", "C"])
    pd.testing.assert_frame_equal(dp.segregate_panel(aggregated_data, segregated_data, match_sum=True),
                                  output_data)
    output_data = pd.DataFrame([[0., 848., 292.],
                                [137., 0., 867.],
                                [508., 969., np.nan],
                                [1273., 137., 326.],
                                [100., 834., 969.],
                                [400., 562., np.nan]], output_index, columns=["A", "B", "C"])
    pd.testing.assert_frame_equal(dp.segregate_panel(aggregated_data, segregated_data, match_sum=False),
                                  output_data)
