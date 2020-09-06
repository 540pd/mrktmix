import mrktmix as mmm
import numpy as np
import pandas as pd

from pytest import approx
from mrktmix.dataprep import mmm_transform as dp


# apl when lag is integer


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
    pd.testing.assert_series_equal(output_single, mmm.apply_apl(df, 0, 1, 0))
    # apply apl on pandas series
    dates = (pd.date_range(start='20180601', end='20180630', freq='W-FRI')
             .strftime('%Y-%m-%d'))
    df = pd.Series(np.array([1., 2., 3., 4., 5.]), index=dates, name="two")
    input_list = [0.0, 1.0, 1.8991444823309347, 2.753418658226993, 3.5561024564142]
    output_single = pd.Series(np.array(input_list), index=dates, name=('two', .50, .70, 1))
    pd.testing.assert_series_equal(output_single, mmm.apply_apl(df, 0.5, .7, 1))


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
