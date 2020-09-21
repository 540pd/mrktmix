import numpy as np
import pandas as pd
from pytest import approx

import mrktmix as mmm
from mrktmix.dataprep import mmm_transform as dp

# apl when lag is integer


def test_create_base_granular():
    # date is list and increasing is False
    df = {pd.Timestamp('2020-03-10 00:00:00'): 1, pd.Timestamp('2020-04-07 00:00:00'): 1}
    df_output = pd.Series(df, name="var1")
    pd.testing.assert_series_equal(df_output, mmm.create_base(
        "var1", ["2020-03-10", "2020-04-07"], "7D", increasing=False, negative=False, panel=None))
    # date is list and increasing is True
    df = {pd.Timestamp('2020-03-10 00:00:00'): 1, pd.Timestamp('2020-04-07 00:00:00'): 2}
    df_output = pd.Series(df, name="var1")
    pd.testing.assert_series_equal(df_output, mmm.create_base(
        "var1", ["2020-03-10", "2020-04-07"], "7D", increasing=True, negative=False, panel=None))
    # date is list and negative is True
    df = {pd.Timestamp('2020-03-10 00:00:00'): -1, pd.Timestamp('2020-04-07 00:00:00'): -2}
    df_output = pd.Series(df, name="var1")
    pd.testing.assert_series_equal(df_output, mmm.create_base(
        "var1", ["2020-03-10", "2020-04-07"], "7D", increasing=True, negative=True, panel=None))
    # date is tuple
    df = {pd.Timestamp('2020-03-10 00:00:00'): 1, pd.Timestamp('2020-03-17 00:00:00'): 1}
    df_output = pd.Series(df, name="var1")
    pd.testing.assert_series_equal(
        df_output,
        mmm.create_base(
            "var1",
            ("2020-03-10",
             "2020-03-21"),
            "7D",
            increasing=False,
            negative=False,
            panel=None))
    # date is string
    df = {pd.Timestamp('2020-03-10 00:00:00'): 1,
          pd.Timestamp('2020-03-17 00:00:00'): 1,
          pd.Timestamp('2020-03-24 00:00:00'): 1}
    df_output = pd.Series(df, name="var1")
    pd.testing.assert_series_equal(
        df_output,
        mmm.create_base(
            "var1",
            "2020-03-10",
            "7D",
            increasing=False,
            negative=False,
            periods=3,
            panel=None))
    # date is list and panel is present
    df_dict = {('panel1', pd.Timestamp('2020-03-10 00:00:00')): 1, ('panel1', pd.Timestamp('2020-04-07 00:00:00')): 1}
    pd.testing.assert_series_equal(pd.Series(df_dict, name="var1"), mmm.create_base(
        "var1", ["2020-03-10", "2020-04-07"], "7D", increasing=False, negative=False, panel="panel1"))


def test_create_base_dataframe():
    df = pd.DataFrame([["var2", ["2020-03-10", "2020-04-07"]],
                       ["var2", ("2020-03-10", "2020-04-07")],
                       ["var3", ("2020-03-10", "2020-04-06")],
                       ["var4", "2020-03-10"]])
    df_dict = {'var2': {('panel1', pd.Timestamp('2020-03-10 00:00:00')): 2.0,
                        ('panel1', pd.Timestamp('2020-03-17 00:00:00')): 1.0,
                        ('panel1', pd.Timestamp('2020-03-24 00:00:00')): 1.0,
                        ('panel1', pd.Timestamp('2020-03-31 00:00:00')): 1.0,
                        ('panel1', pd.Timestamp('2020-04-07 00:00:00')): 2.0,
                        ('panel2', pd.Timestamp('2020-03-10 00:00:00')): None,
                        ('panel2', pd.Timestamp('2020-03-17 00:00:00')): None,
                        ('panel2', pd.Timestamp('2020-03-24 00:00:00')): None,
                        ('panel2', pd.Timestamp('2020-03-31 00:00:00')): None},
               'var3': {('panel1', pd.Timestamp('2020-03-10 00:00:00')): None,
                        ('panel1', pd.Timestamp('2020-03-17 00:00:00')): None,
                        ('panel1', pd.Timestamp('2020-03-24 00:00:00')): None,
                        ('panel1', pd.Timestamp('2020-03-31 00:00:00')): None,
                        ('panel1', pd.Timestamp('2020-04-07 00:00:00')): None,
                        ('panel2', pd.Timestamp('2020-03-10 00:00:00')): 1.0,
                        ('panel2', pd.Timestamp('2020-03-17 00:00:00')): 1.0,
                        ('panel2', pd.Timestamp('2020-03-24 00:00:00')): 1.0,
                        ('panel2', pd.Timestamp('2020-03-31 00:00:00')): 1.0},
               'var4': {('panel1', pd.Timestamp('2020-03-10 00:00:00')): 1.0,
                        ('panel1', pd.Timestamp('2020-03-17 00:00:00')): 1.0,
                        ('panel1', pd.Timestamp('2020-03-24 00:00:00')): 1.0,
                        ('panel1', pd.Timestamp('2020-03-31 00:00:00')): 1.0,
                        ('panel1', pd.Timestamp('2020-04-07 00:00:00')): 1.0,
                        ('panel2', pd.Timestamp('2020-03-10 00:00:00')): None,
                        ('panel2', pd.Timestamp('2020-03-17 00:00:00')): None,
                        ('panel2', pd.Timestamp('2020-03-24 00:00:00')): None,
                        ('panel2', pd.Timestamp('2020-03-31 00:00:00')): None}}
    pd.testing.assert_frame_equal(
        pd.DataFrame(df_dict).fillna(0),
        mmm.create_base(
            df[0].values,
            df[1].values,
            "7D",
            increasing=False,
            negative=False,
            periods=5,
            panel=[
                "panel1",
                "panel1",
                "panel2",
                "panel1"]).fillna(0))


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


def test_apply_apl():
    # apply apl on pandas series
    panel_var = {False: [('Intercept', 0, 1, 0), ('one', 0, 1, 0), ('two', 0, 1, 0), ('one', 0, 1, 1)]}
    df_1 = pd.DataFrame({'two': [1., 2., 3., 4.], 'one': [4., 3., 2., 1.], 'Intercept': [4., 3., 2., 1.]},
                        index=['2', '5', '7', '9'])
    expected_output = {('Intercept', 0, 1, 0): {'2': 4.0, '5': 3.0, '7': 2.0, '9': 1.0},
                       ('one', 0, 1, 0): {'2': 4.0, '5': 3.0, '7': 2.0, '9': 1.0},
                       ('two', 0, 1, 0): {'2': 1.0, '5': 2.0, '7': 3.0, '9': 4.0},
                       ('one', 0, 1, 1): {'2': 0.0, '5': 4.0, '7': 3.0, '9': 2.0}}
    expected_output = pd.DataFrame(expected_output)
    expected_output.columns.names = ['Variable', 'Adstock', 'Power', 'Lag']
    pd.testing.assert_frame_equal(mmm.apply_apl(df_1, panel_var), pd.DataFrame(expected_output))
    # apply apl on pandas series
    panel_var = {"METRO": [('A', 0, 1, 0), ('B', 0, 1, 0), ('A', 0, 1, 1)], "REGIONAL": [('A', 0, 1, 2), ('A', 0, 1, 1)]}
    df_ind = [np.array(["METRO", "METRO", "METRO", "METRO", "METRO", "REGIONAL", "REGIONAL", "REGIONAL", "REGIONAL", "REGIONAL",
                        "VILLAGE", "VILLAGE", "VILLAGE", "VILLAGE", "VILLAGE"]),
              np.array(["1/1/2018", "2/1/2018", "3/1/2018", "4/1/2018", "5/1/2018", "1/1/2018", "2/1/2018", "3/1/2018", "4/1/2018",
                        "5/1/2018", "1/1/2018", "2/1/2018", "3/1/2018", "4/1/2018", "5/1/2018"])]
    df_2 = pd.DataFrame({'A': [773, 137, 508, 562, 365, 500, 100, 400, 79, 365, 773, 137, 508, 562, 365],
                         'B': [848, 326, 969, 730, 761, 137, 508, 562, 365, 761, 848, 326, 969, 730, 761],
                         'C': [292, 867, 323, 910, 729, 326, 969, 730, 761, 729, 292, 867, 323, 910, 729],
                         'Dep': [479532.,  85700., 307530., 347654., 229367.,  92130., 99820., 103480.,  97567.5, 105322., 104699.,
                                 36128., 110005., 89533.,  85296.]}, index=df_ind)
    expected_output = {('A', 0, 1, 0): {('METRO', '1/1/2018'): 773.0,
                                        ('METRO', '2/1/2018'): 137.0,
                                        ('METRO', '3/1/2018'): 508.0,
                                        ('METRO', '4/1/2018'): 562.0,
                                        ('METRO', '5/1/2018'): 365.0,
                                        ('REGIONAL', '1/1/2018'): np.nan,
                                        ('REGIONAL', '2/1/2018'): np.nan,
                                        ('REGIONAL', '3/1/2018'): np.nan,
                                        ('REGIONAL', '4/1/2018'): np.nan,
                                        ('REGIONAL', '5/1/2018'): np.nan},
                       ('A', 0, 1, 1): {('METRO', '1/1/2018'): 0.0,
                                        ('METRO', '2/1/2018'): 773.0,
                                        ('METRO', '3/1/2018'): 137.0,
                                        ('METRO', '4/1/2018'): 508.0,
                                        ('METRO', '5/1/2018'): 562.0,
                                        ('REGIONAL', '1/1/2018'): 0.0,
                                        ('REGIONAL', '2/1/2018'): 500.0,
                                        ('REGIONAL', '3/1/2018'): 100.0,
                                        ('REGIONAL', '4/1/2018'): 400.0,
                                        ('REGIONAL', '5/1/2018'): 79.0},
                       ('A', 0, 1, 2): {('METRO', '1/1/2018'): np.nan,
                                        ('METRO', '2/1/2018'): np.nan,
                                        ('METRO', '3/1/2018'): np.nan,
                                        ('METRO', '4/1/2018'): np.nan,
                                        ('METRO', '5/1/2018'): np.nan,
                                        ('REGIONAL', '1/1/2018'): 0.0,
                                        ('REGIONAL', '2/1/2018'): 0.0,
                                        ('REGIONAL', '3/1/2018'): 500.0,
                                        ('REGIONAL', '4/1/2018'): 100.0,
                                        ('REGIONAL', '5/1/2018'): 400.0},
                       ('B', 0, 1, 0): {('METRO', '1/1/2018'): 848.0,
                                        ('METRO', '2/1/2018'): 326.0,
                                        ('METRO', '3/1/2018'): 969.0,
                                        ('METRO', '4/1/2018'): 730.0,
                                        ('METRO', '5/1/2018'): 761.0,
                                        ('REGIONAL', '1/1/2018'): np.nan,
                                        ('REGIONAL', '2/1/2018'): np.nan,
                                        ('REGIONAL', '3/1/2018'): np.nan,
                                        ('REGIONAL', '4/1/2018'): np.nan,
                                        ('REGIONAL', '5/1/2018'): np.nan}}
    expected_output = pd.DataFrame(expected_output)
    expected_output.columns.names = ['Variable', 'Adstock', 'Power', 'Lag']
    pd.testing.assert_frame_equal(mmm.apply_apl(df_2, panel_var).fillna(0), expected_output.fillna(0))


def test_apply_coef():
    # without dependent
    coef_ind = [["METRO", "METRO", "METRO", "METRO", "REGIONAL", "REGIONAL", "REGIONAL", "REGIONAL", "VILLAGE", "VILLAGE", "VILLAGE",
                 "VILLAGE"], [("Intercept", 0, 1, 0), ("A", 0, 1, 0), ("B", 0, 1, 0), ("D", 0, 1, 0), ("Intercept", 0, 1, 0),
                              ("A", 0, 1, 0), ("B", 0, 1, 0), ("D", 0, 1, 0), ("Intercept", 0, 1, 0), ("A", 0, 1, 0), ("B", 0, 1, 0),
                              ("D", 0, 1, 0)]]
    coef_2 = pd.Series(data=[3240, 600, 10, 0.9, 90000, 0.6, 20, 0.9, 500, 25, 100, 0.9], index=coef_ind)
    df_ind = [np.array(["METRO", "METRO", "METRO", "METRO", "METRO", "REGIONAL", "REGIONAL", "REGIONAL", "REGIONAL", "REGIONAL", "VILLAGE",
                        "VILLAGE", "VILLAGE", "VILLAGE", "VILLAGE"]),
              np.array(["1/1/2018", "2/1/2018", "3/1/2018", "4/1/2018", "5/1/2018", "1/1/2018", "2/1/2018", "3/1/2018", "4/1/2018",
                        "5/1/2018", "1/1/2018", "2/1/2018", "3/1/2018", "4/1/2018", "5/1/2018"])]
    df_2 = pd.DataFrame({'A': [773, 137, 508, 562, 365, 500, 100, 400, 79, 365, 773, 137, 508, 562, 365],
                         'B': [848, 326, 969, 730, 761, 137, 508, 562, 365, 761, 848, 326, 969, 730, 761],
                         'C': [292, 867, 323, 910, 729, 326, 969, 730, 761, 729, 292, 867, 323, 910, 729],
                         'D': [479532.,  85700., 307530., 347654., 229367.,  92130., 99820., 103480.,  97567.5, 105322., 104699.,  36128.,
                               110005., 89533.,  85296.]}, index=df_ind)
    df_2["Intercept"] = 1
    expected_output = {('A', 0, 1, 0): {('METRO', '1/1/2018'): 463800.0,
                                        ('METRO', '2/1/2018'): 82200.0,
                                        ('METRO', '3/1/2018'): 304800.0,
                                        ('METRO', '4/1/2018'): 337200.0,
                                        ('METRO', '5/1/2018'): 219000.0,
                                        ('REGIONAL', '1/1/2018'): 300.0,
                                        ('REGIONAL', '2/1/2018'): 60.0,
                                        ('REGIONAL', '3/1/2018'): 240.0,
                                        ('REGIONAL', '4/1/2018'): 47.4,
                                        ('REGIONAL', '5/1/2018'): 219.0,
                                        ('VILLAGE', '1/1/2018'): 19325.0,
                                        ('VILLAGE', '2/1/2018'): 3425.0,
                                        ('VILLAGE', '3/1/2018'): 12700.0,
                                        ('VILLAGE', '4/1/2018'): 14050.0,
                                        ('VILLAGE', '5/1/2018'): 9125.0},
                       ('B', 0, 1, 0): {('METRO', '1/1/2018'): 8480.0,
                                        ('METRO', '2/1/2018'): 3260.0,
                                        ('METRO', '3/1/2018'): 9690.0,
                                        ('METRO', '4/1/2018'): 7300.0,
                                        ('METRO', '5/1/2018'): 7610.0,
                                        ('REGIONAL', '1/1/2018'): 2740.0,
                                        ('REGIONAL', '2/1/2018'): 10160.0,
                                        ('REGIONAL', '3/1/2018'): 11240.0,
                                        ('REGIONAL', '4/1/2018'): 7300.0,
                                        ('REGIONAL', '5/1/2018'): 15220.0,
                                        ('VILLAGE', '1/1/2018'): 84800.0,
                                        ('VILLAGE', '2/1/2018'): 32600.0,
                                        ('VILLAGE', '3/1/2018'): 96900.0,
                                        ('VILLAGE', '4/1/2018'): 73000.0,
                                        ('VILLAGE', '5/1/2018'): 76100.0},
                       ('D', 0, 1, 0): {('METRO', '1/1/2018'): 431578.8,
                                        ('METRO', '2/1/2018'): 77130.0,
                                        ('METRO', '3/1/2018'): 276777.0,
                                        ('METRO', '4/1/2018'): 312888.60000000003,
                                        ('METRO', '5/1/2018'): 206430.30000000002,
                                        ('REGIONAL', '1/1/2018'): 82917.0,
                                        ('REGIONAL', '2/1/2018'): 89838.0,
                                        ('REGIONAL', '3/1/2018'): 93132.0,
                                        ('REGIONAL', '4/1/2018'): 87810.75,
                                        ('REGIONAL', '5/1/2018'): 94789.8,
                                        ('VILLAGE', '1/1/2018'): 94229.1,
                                        ('VILLAGE', '2/1/2018'): 32515.2,
                                        ('VILLAGE', '3/1/2018'): 99004.5,
                                        ('VILLAGE', '4/1/2018'): 80579.7,
                                        ('VILLAGE', '5/1/2018'): 76766.40000000001},
                       ('Intercept', 0, 1, 0): {('METRO', '1/1/2018'): 3240.0,
                                                ('METRO', '2/1/2018'): 3240.0,
                                                ('METRO', '3/1/2018'): 3240.0,
                                                ('METRO', '4/1/2018'): 3240.0,
                                                ('METRO', '5/1/2018'): 3240.0,
                                                ('REGIONAL', '1/1/2018'): 90000.0,
                                                ('REGIONAL', '2/1/2018'): 90000.0,
                                                ('REGIONAL', '3/1/2018'): 90000.0,
                                                ('REGIONAL', '4/1/2018'): 90000.0,
                                                ('REGIONAL', '5/1/2018'): 90000.0,
                                                ('VILLAGE', '1/1/2018'): 500.0,
                                                ('VILLAGE', '2/1/2018'): 500.0,
                                                ('VILLAGE', '3/1/2018'): 500.0,
                                                ('VILLAGE', '4/1/2018'): 500.0,
                                                ('VILLAGE', '5/1/2018'): 500.0}}
    expected_output = pd.DataFrame(expected_output)
    expected_output.columns.names = ["Variable", "Adstock", "Power", "Lag"]
    pd.testing.assert_frame_equal(mmm.apply_coef(df_2, coef_2, None), expected_output)
    # with dependent
    coef_ind = [["METRO", "METRO", "METRO", "METRO", "REGIONAL", "REGIONAL", "REGIONAL", "REGIONAL", "VILLAGE", "VILLAGE", "VILLAGE",
                 "VILLAGE"],
                [("Intercept", 0, 1, 0), ("A", 0, 1, 0), ("B", 0, 1, 0), ("D", 0, 1, 0), ("Intercept", 0, 1, 0), ("A", 0, 1, 0),
                 ("B", 0, 1, 0), ("D", 0, 1, 0), ("Intercept", 0, 1, 0), ("A", 0, 1, 0), ("B", 0, 1, 0), ("D", 0, 1, 0)]]
    coef_2 = pd.Series(data=[3240, 600, 10, 0.9, 90000, 0.6, 20, 0.9, 500, 25, 100, 0.9], index=coef_ind)
    df_ind = [np.array(["METRO", "METRO", "METRO", "METRO", "METRO", "REGIONAL", "REGIONAL", "REGIONAL", "REGIONAL", "REGIONAL", "VILLAGE",
                        "VILLAGE", "VILLAGE", "VILLAGE", "VILLAGE"]),
              np.array(["1/1/2018", "2/1/2018", "3/1/2018", "4/1/2018", "5/1/2018", "1/1/2018",
                        "2/1/2018", "3/1/2018", "4/1/2018", "5/1/2018", "1/1/2018", "2/1/2018", "3/1/2018", "4/1/2018", "5/1/2018"])]
    df_2 = pd.DataFrame({'A': [773, 137, 508, 562, 365, 500, 100, 400, 79, 365, 773, 137, 508, 562, 365],
                         'B': [848, 326, 969, 730, 761, 137, 508, 562, 365, 761, 848, 326, 969, 730, 761],
                         'C': [292, 867, 323, 910, 729, 326, 969, 730, 761, 729, 292, 867, 323, 910, 729],
                         'D': [479532.,  85700., 307530., 347654., 229367.,  92130., 99820., 103480.,  97567.5, 105322., 104699.,  36128.,
                               110005., 89533., 85296.]}, index=df_ind)
    df_2["Intercept"] = 1
    dep = pd.DataFrame({'Dep': [773, 137, 508, 562, 365, 500, 100, 400, 79, 365, 773, 137, 508, 562, 365]}, index=df_ind)
    expected_output = {('A', 0, 1, 0): {('METRO', '1/1/2018'): 463800.0,
                                        ('METRO', '2/1/2018'): 82200.0,
                                        ('METRO', '3/1/2018'): 304800.0,
                                        ('METRO', '4/1/2018'): 337200.0,
                                        ('METRO', '5/1/2018'): 219000.0,
                                        ('REGIONAL', '1/1/2018'): 300.0,
                                        ('REGIONAL', '2/1/2018'): 60.0,
                                        ('REGIONAL', '3/1/2018'): 240.0,
                                        ('REGIONAL', '4/1/2018'): 47.4,
                                        ('REGIONAL', '5/1/2018'): 219.0,
                                        ('VILLAGE', '1/1/2018'): 19325.0,
                                        ('VILLAGE', '2/1/2018'): 3425.0,
                                        ('VILLAGE', '3/1/2018'): 12700.0,
                                        ('VILLAGE', '4/1/2018'): 14050.0,
                                        ('VILLAGE', '5/1/2018'): 9125.0},
                       ('B', 0, 1, 0): {('METRO', '1/1/2018'): 8480.0,
                                        ('METRO', '2/1/2018'): 3260.0,
                                        ('METRO', '3/1/2018'): 9690.0,
                                        ('METRO', '4/1/2018'): 7300.0,
                                        ('METRO', '5/1/2018'): 7610.0,
                                        ('REGIONAL', '1/1/2018'): 2740.0,
                                        ('REGIONAL', '2/1/2018'): 10160.0,
                                        ('REGIONAL', '3/1/2018'): 11240.0,
                                        ('REGIONAL', '4/1/2018'): 7300.0,
                                        ('REGIONAL', '5/1/2018'): 15220.0,
                                        ('VILLAGE', '1/1/2018'): 84800.0,
                                        ('VILLAGE', '2/1/2018'): 32600.0,
                                        ('VILLAGE', '3/1/2018'): 96900.0,
                                        ('VILLAGE', '4/1/2018'): 73000.0,
                                        ('VILLAGE', '5/1/2018'): 76100.0},
                       ('D', 0, 1, 0): {('METRO', '1/1/2018'): 431578.8,
                                        ('METRO', '2/1/2018'): 77130.0,
                                        ('METRO', '3/1/2018'): 276777.0,
                                        ('METRO', '4/1/2018'): 312888.60000000003,
                                        ('METRO', '5/1/2018'): 206430.30000000002,
                                        ('REGIONAL', '1/1/2018'): 82917.0,
                                        ('REGIONAL', '2/1/2018'): 89838.0,
                                        ('REGIONAL', '3/1/2018'): 93132.0,
                                        ('REGIONAL', '4/1/2018'): 87810.75,
                                        ('REGIONAL', '5/1/2018'): 94789.8,
                                        ('VILLAGE', '1/1/2018'): 94229.1,
                                        ('VILLAGE', '2/1/2018'): 32515.2,
                                        ('VILLAGE', '3/1/2018'): 99004.5,
                                        ('VILLAGE', '4/1/2018'): 80579.7,
                                        ('VILLAGE', '5/1/2018'): 76766.40000000001},
                       ('Intercept', 0, 1, 0): {('METRO', '1/1/2018'): 3240.0,
                                                ('METRO', '2/1/2018'): 3240.0,
                                                ('METRO', '3/1/2018'): 3240.0,
                                                ('METRO', '4/1/2018'): 3240.0,
                                                ('METRO', '5/1/2018'): 3240.0,
                                                ('REGIONAL', '1/1/2018'): 90000.0,
                                                ('REGIONAL', '2/1/2018'): 90000.0,
                                                ('REGIONAL', '3/1/2018'): 90000.0,
                                                ('REGIONAL', '4/1/2018'): 90000.0,
                                                ('REGIONAL', '5/1/2018'): 90000.0,
                                                ('VILLAGE', '1/1/2018'): 500.0,
                                                ('VILLAGE', '2/1/2018'): 500.0,
                                                ('VILLAGE', '3/1/2018'): 500.0,
                                                ('VILLAGE', '4/1/2018'): 500.0,
                                                ('VILLAGE', '5/1/2018'): 500.0},
                       ('Residual', 0, 1, 0): {('METRO', '1/1/2018'): -906325.8,
                                               ('METRO', '2/1/2018'): -165693.0,
                                               ('METRO', '3/1/2018'): -593999.0,
                                               ('METRO', '4/1/2018'): -660066.6000000001,
                                               ('METRO', '5/1/2018'): -435915.30000000005,
                                               ('REGIONAL', '1/1/2018'): -175457.0,
                                               ('REGIONAL', '2/1/2018'): -189958.0,
                                               ('REGIONAL', '3/1/2018'): -194212.0,
                                               ('REGIONAL', '4/1/2018'): -185079.15,
                                               ('REGIONAL', '5/1/2018'): -199863.8,
                                               ('VILLAGE', '1/1/2018'): -198081.1,
                                               ('VILLAGE', '2/1/2018'): -68903.2,
                                               ('VILLAGE', '3/1/2018'): -208596.5,
                                               ('VILLAGE', '4/1/2018'): -167567.7,
                                               ('VILLAGE', '5/1/2018'): -162126.40000000002}}
    expected_output = pd.DataFrame(expected_output)
    expected_output.columns.names = ["Variable", "Adstock", "Power", "Lag"]
    pd.testing.assert_frame_equal(mmm.apply_coef(df_2, coef_2, dep["Dep"]), expected_output)


def test_collapse_date():
    # on dataframe
    panel_var = {"METRO": [('A', 0, 1, 0), ('B', 0, 1, 0), ('A', 0, 1, 1)], "REGIONAL": [('A', 0, 1, 2), ('A', 0, 1, 1)]}
    df_ind = [np.array(["METRO", "METRO", "METRO", "METRO", "METRO", "REGIONAL", "REGIONAL", "REGIONAL", "REGIONAL", "REGIONAL",
                        "VILLAGE", "VILLAGE", "VILLAGE", "VILLAGE", "VILLAGE"]),
              np.array(["1/1/2018", "2/1/2018", "3/1/2018", "4/1/2018", "5/1/2018", "1/1/2018", "2/1/2018", "3/1/2018", "4/1/2018",
                        "5/1/2018", "1/1/2018", "2/1/2018", "3/1/2018", "4/1/2018", "5/1/2018"])]
    df_2 = pd.DataFrame({'A': [773, 137, 508, 562, 365, 500, 100, 400, 79, 365, 773, 137, 508, 562, 365],
                         'B': [848, 326, 969, 730, 761, 137, 508, 562, 365, 761, 848, 326, 969, 730, 761],
                         'C': [292, 867, 323, 910, 729, 326, 969, 730, 761, 729, 292, 867, 323, 910, 729],
                         'Dep': [479532.,  85700., 307530., 347654., 229367.,  92130., 99820., 103480.,  97567.5, 105322., 104699.,
                                 36128., 110005., 89533.,  85296.]}, index=df_ind)
    date_dict = {'test Year': ('1/1/2018', '4/1/2018'), 'train_year': ('1/1/2018', '5/1/2018')}
    expected_output = {'test Year': {('METRO', 'A'): 1980.0,
                                     ('METRO', 'B'): 2873.0,
                                     ('METRO', 'C'): 2392.0,
                                     ('METRO', 'Dep'): 1220416.0,
                                     ('REGIONAL', 'A'): 1079.0,
                                     ('REGIONAL', 'B'): 1572.0,
                                     ('REGIONAL', 'C'): 2786.0,
                                     ('REGIONAL', 'Dep'): 392997.5,
                                     ('VILLAGE', 'A'): 1980.0,
                                     ('VILLAGE', 'B'): 2873.0,
                                     ('VILLAGE', 'C'): 2392.0,
                                     ('VILLAGE', 'Dep'): 340365.0},
                       'train_year': {('METRO', 'A'): 2345.0,
                                      ('METRO', 'B'): 3634.0,
                                      ('METRO', 'C'): 3121.0,
                                      ('METRO', 'Dep'): 1449783.0,
                                      ('REGIONAL', 'A'): 1444.0,
                                      ('REGIONAL', 'B'): 2333.0,
                                      ('REGIONAL', 'C'): 3515.0,
                                      ('REGIONAL', 'Dep'): 498319.5,
                                      ('VILLAGE', 'A'): 2345.0,
                                      ('VILLAGE', 'B'): 3634.0,
                                      ('VILLAGE', 'C'): 3121.0,
                                      ('VILLAGE', 'Dep'): 425661.0}}
    expected_output = pd.DataFrame(expected_output)
    pd.testing.assert_frame_equal(mmm.collapse_date(df_2, date_dict), expected_output)
    # on dataframe with multiindex column
    panel_var = {"METRO": [('A', 0, 1, 0), ('B', 0, 1, 0), ('A', 0, 1, 1)], "REGIONAL": [('A', 0, 1, 2), ('A', 0, 1, 1)]}
    dep_decompose = mmm.apply_apl(df_2, panel_var)
    date_dict = {'test Year': ('1/1/2018', '4/1/2018'), 'train_year': ('1/1/2018', '5/1/2018')}
    expected_output = {'test Year': {('METRO', 'A', 0, 1, 0): 1980.0,
                                     ('METRO', 'A', 0, 1, 1): 1418.0,
                                     ('METRO', 'A', 0, 1, 2): 0.0,
                                     ('METRO', 'B', 0, 1, 0): 2873.0,
                                     ('REGIONAL', 'A', 0, 1, 0): 0.0,
                                     ('REGIONAL', 'A', 0, 1, 1): 1000.0,
                                     ('REGIONAL', 'A', 0, 1, 2): 600.0,
                                     ('REGIONAL', 'B', 0, 1, 0): 0.0},
                       'train_year': {('METRO', 'A', 0, 1, 0): 2345.0,
                                      ('METRO', 'A', 0, 1, 1): 1980.0,
                                      ('METRO', 'A', 0, 1, 2): 0.0,
                                      ('METRO', 'B', 0, 1, 0): 3634.0,
                                      ('REGIONAL', 'A', 0, 1, 0): 0.0,
                                      ('REGIONAL', 'A', 0, 1, 1): 1079.0,
                                      ('REGIONAL', 'A', 0, 1, 2): 1000.0,
                                      ('REGIONAL', 'B', 0, 1, 0): 0.0}}
    expected_output = pd.DataFrame(expected_output)
    expected_output.index.names = [None, "Variable", "Adstock", "Power", "Lag"]
    pd.testing.assert_frame_equal(mmm.collapse_date(dep_decompose, date_dict), expected_output)
