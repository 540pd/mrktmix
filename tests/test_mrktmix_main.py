import random

import numpy as np
import pandas as pd

import mrktmix as mmm


def test_create_mdldata():
    # Test simple modeling data
    input_df1 = {'Panel': {0: 'panel1',
                           1: 'panel1',
                           2: 'panel2',
                           3: 'panel2',
                           4: 'panel2',
                           5: 'panel3',
                           6: 'panel3'},
                 'Channel': {0: 'TV_Free',
                             1: 'TV_Free',
                             2: 'TV_Free',
                             3: 'TV_Free',
                             4: 'TV_Free',
                             5: 'TV_Free',
                             6: 'TV_Free'},
                 'Metric': {0: 'GRp',
                            1: 'Spend',
                            2: 'GRP',
                            3: 'SpenD',
                            4: 'Spend',
                            5: 'gRP',
                            6: 'gRP'},
                 'Metric_Value': {0: 33, 1: 102, 2: 45, 3: 129, 4: 170, 5: 24, 6: 49}}
    input_df1 = pd.DataFrame.from_dict(input_df1)
    input_df2 = {'Panel': {0: 'panel1',
                           1: 'panel1',
                           2: 'panel2',
                           3: 'panel2',
                           4: 'panel2',
                           5: 'panel3',
                           6: 'panel3'},
                 'Channel': {0: 'TV_Free',
                             1: 'TV_Free',
                             2: 'TV_Free',
                             3: 'TV_Free',
                             4: 'TV_Free',
                             5: 'TV_Free',
                             6: 'TV_Free'},
                 'Metric': {0: 'GRp',
                            1: 'Spend',
                            2: 'GRP',
                            3: 'SpenD',
                            4: 'Spend',
                            5: 'gRP',
                            6: 'gRP'},
                 'Metric_Value': {0: 33, 1: 102, 2: 45, 3: 129, 4: 170, 5: 24, 6: 49}}
    input_df2 = pd.DataFrame.from_dict(input_df2)
    input_files = {"source1": input_df1, "source2": input_df2}

    # Simple case
    output_mapping = {'GRP': 'F85',
                      'GRp': 'F85',
                      'SpenD': 'GMJ',
                      'Spend': 'GMJ',
                      'TV_Free': 'ZOZ',
                      'gRP': 'F85'}
    output_data = {('sum', 'ZOZ_F85'): {'panel1': 66.0, 'panel2': 90.0, 'panel3': 146.0},
                   ('sum', 'ZOZ_GMJ'): {'panel1': 204.0, 'panel2': 598.0, 'panel3': np.nan}}
    output_data = pd.DataFrame.from_dict(output_data)
    output_data.columns.names = ["_summary_type_", '_Variable_']
    output_data.index.names = ['Panel']
    file_map = pd.DataFrame(np.array([['TV_Free', 'GRp'],
                                      ['TV_Free', 'Spend'],
                                      ['TV_Free', 'GRP'],
                                      ['TV_Free', 'SpenD'],
                                      ['TV_Free', 'gRP'],
                                      ['TV_Free', 'GRp'],
                                      ['TV_Free', 'Spend'],
                                      ['TV_Free', 'GRP'],
                                      ['TV_Free', 'SpenD'],
                                      ['TV_Free', 'gRP']]),
                            index=pd.MultiIndex.from_tuples([('source1', 'ZOZ_F85'),
                                                             ('source1', 'ZOZ_GMJ'),
                                                             ('source1', 'ZOZ_F85'),
                                                             ('source1', 'ZOZ_GMJ'),
                                                             ('source1', 'ZOZ_F85'),
                                                             ('source2', 'ZOZ_F85'),
                                                             ('source2', 'ZOZ_GMJ'),
                                                             ('source2', 'ZOZ_F85'),
                                                             ('source2', 'ZOZ_GMJ'),
                                                             ('source2', 'ZOZ_F85')],
                                                            names=['_File_', '_Variable_']),
                            columns=["Channel", "Metric"])
    random.seed(999)
    mdl_data, code_mapping, file_mapping = mmm.create_mdldata(input_files, ['Panel'], ["Channel", "Metric"], "Metric_Value")
    pd.testing.assert_frame_equal(mdl_data, output_data)
    assert code_mapping == output_mapping
    pd.testing.assert_frame_equal(file_mapping, file_map)

    # Test when input mapping is given
    output_data = {('sum', 'TV_GRP'): {'panel1': 66.0, 'panel2': 90.0, 'panel3': 146.0},
                   ('sum', 'TV_SPD'): {'panel1': 204.0, 'panel2': 598.0, 'panel3': np.nan}}
    output_data = pd.DataFrame.from_dict(output_data)
    output_data.columns.names = ["_summary_type_", '_Variable_']
    output_data.index.names = ['Panel']
    output_mapping = {'GRP': 'GRP',
                      'Spend': 'SPD',
                      'TV_Free': 'TV',
                      'GRp': 'GRP',
                      'SpenD': 'SPD',
                      'gRP': 'GRP'}
    file_map = pd.DataFrame(np.array([['TV_Free', 'GRp'],
                                      ['TV_Free', 'Spend'],
                                      ['TV_Free', 'GRP'],
                                      ['TV_Free', 'SpenD'],
                                      ['TV_Free', 'gRP'],
                                      ['TV_Free', 'GRp'],
                                      ['TV_Free', 'Spend'],
                                      ['TV_Free', 'GRP'],
                                      ['TV_Free', 'SpenD'],
                                      ['TV_Free', 'gRP']]),
                            index=pd.MultiIndex.from_tuples([('source1', 'TV_GRP'),
                                                             ('source1', 'TV_SPD'),
                                                             ('source1', 'TV_GRP'),
                                                             ('source1', 'TV_SPD'),
                                                             ('source1', 'TV_GRP'),
                                                             ('source2', 'TV_GRP'),
                                                             ('source2', 'TV_SPD'),
                                                             ('source2', 'TV_GRP'),
                                                             ('source2', 'TV_SPD'),
                                                             ('source2', 'TV_GRP')],
                                                            names=['_File_', '_Variable_']),
                            columns=["Channel", "Metric"])
    random.seed(999)
    mdl_data, code_mapping, file_mapping = mmm.create_mdldata(
        input_files, ['Panel'], [
            "Channel", "Metric"], "Metric_Value", description2code={
            'GRP': 'GRP', "Spend": 'SPD', "TV_Free": "TV"})
    pd.testing.assert_frame_equal(mdl_data, output_data)
    assert code_mapping == output_mapping
    pd.testing.assert_frame_equal(file_mapping, file_map)

    # case sensitive
    output_data = {('sum', 'TV_4IU'): {'panel1': np.nan, 'panel2': 258.0, 'panel3': np.nan},
                   ('sum', 'TV_F85'): {'panel1': 66.0, 'panel2': np.nan, 'panel3': np.nan},
                   ('sum', 'TV_GMJ'): {'panel1': np.nan, 'panel2': np.nan, 'panel3': 146.0},
                   ('sum', 'TV_GRP'): {'panel1': np.nan, 'panel2': 90.0, 'panel3': np.nan},
                   ('sum', 'TV_SPD'): {'panel1': 204.0, 'panel2': 340.0, 'panel3': np.nan}}
    output_data = pd.DataFrame.from_dict(output_data)
    output_data.columns.names = ["_summary_type_", '_Variable_']
    output_data.index.names = ['Panel']
    output_mapping = {'GRP': 'GRP',
                      'Spend': 'SPD',
                      'TV_Free': 'TV',
                      'GRp': 'F85',
                      'SpenD': '4IU',
                      'gRP': 'GMJ'}
    file_map = pd.DataFrame(np.array([['TV_Free', 'GRp'],
                                      ['TV_Free', 'Spend'],
                                      ['TV_Free', 'GRP'],
                                      ['TV_Free', 'SpenD'],
                                      ['TV_Free', 'gRP'],
                                      ['TV_Free', 'GRp'],
                                      ['TV_Free', 'Spend'],
                                      ['TV_Free', 'GRP'],
                                      ['TV_Free', 'SpenD'],
                                      ['TV_Free', 'gRP']]),
                            index=pd.MultiIndex.from_tuples([('source1', 'TV_F85'),
                                                             ('source1', 'TV_SPD'),
                                                             ('source1', 'TV_GRP'),
                                                             ('source1', 'TV_4IU'),
                                                             ('source1', 'TV_GMJ'),
                                                             ('source2', 'TV_F85'),
                                                             ('source2', 'TV_SPD'),
                                                             ('source2', 'TV_GRP'),
                                                             ('source2', 'TV_4IU'),
                                                             ('source2', 'TV_GMJ')],
                                                            names=['_File_', '_Variable_']),
                            columns=["Channel", "Metric"])
    # case sensitive
    random.seed(999)
    mdl_data, code_mapping, file_mapping = mmm.create_mdldata(
        input_files, ['Panel'], ["Channel", "Metric"], "Metric_Value",
        description2code={'GRP': 'GRP', "Spend": 'SPD', "TV_Free": "TV"},
        case_sensitive=True, iteratively=False)
    pd.testing.assert_frame_equal(mdl_data, output_data)
    assert code_mapping == output_mapping
    pd.testing.assert_frame_equal(file_mapping, file_map)

    # case sensitive and iteratively
    output = pd.DataFrame(np.array([[np.nan, 33., np.nan, np.nan, 102., np.nan, 33., np.nan, np.nan, 102.],
                                    [129., np.nan, np.nan, 45., 170., 129., np.nan, np.nan, 45., 170.],
                                    [np.nan, np.nan, 73., np.nan, np.nan, np.nan, np.nan, 73., np.nan, np.nan]]),
                          ['panel1', 'panel2', 'panel3'],
                          pd.MultiIndex.from_tuples([('sum', 'TV_4IU'),
                                                     ('sum', 'TV_F85'),
                                                     ('sum', 'TV_GMJ'),
                                                     ('sum', 'TV_GRP'),
                                                     ('sum', 'TV_SPD'),
                                                     ('sum', 'TV_4IU'),
                                                     ('sum', 'TV_F85'),
                                                     ('sum', 'TV_GMJ'),
                                                     ('sum', 'TV_GRP'),
                                                     ('sum', 'TV_SPD')],
                                                    names=('_summary_type_', '_Variable_')))
    output.index.names = ['Panel']
    output_mapping = {'GRP': 'GRP',
                      'Spend': 'SPD',
                      'TV_Free': 'TV',
                      'GRp': 'F85',
                      'SpenD': '4IU',
                      'gRP': 'GMJ'}
    file_map = pd.DataFrame(np.array([['TV_Free', 'GRp'],
                                      ['TV_Free', 'Spend'],
                                      ['TV_Free', 'GRP'],
                                      ['TV_Free', 'SpenD'],
                                      ['TV_Free', 'gRP'],
                                      ['TV_Free', 'GRp'],
                                      ['TV_Free', 'Spend'],
                                      ['TV_Free', 'GRP'],
                                      ['TV_Free', 'SpenD'],
                                      ['TV_Free', 'gRP']]),
                            pd.MultiIndex.from_tuples([('source1', 'TV_F85'),
                                                       ('source1', 'TV_SPD'),
                                                       ('source1', 'TV_GRP'),
                                                       ('source1', 'TV_4IU'),
                                                       ('source1', 'TV_GMJ'),
                                                       ('source2', 'TV_F85'),
                                                       ('source2', 'TV_SPD'),
                                                       ('source2', 'TV_GRP'),
                                                       ('source2', 'TV_4IU'),
                                                       ('source2', 'TV_GMJ')],
                                                      names=('_File_', '_Variable_')),
                            columns=["Channel", "Metric"])
    random.seed(999)
    mdl_data, code_mapping, file_mapping = mmm.create_mdldata(
        input_files, ["Panel"], ["Channel", "Metric"], "Metric_Value",
        description2code={'GRP': 'GRP', "Spend": 'SPD', "TV_Free": "TV"},
        case_sensitive=True, iteratively=True)
    pd.testing.assert_frame_equal(mdl_data, output)
    assert code_mapping == output_mapping
    pd.testing.assert_frame_equal(file_mapping, file_map)


def test_spread_notna():
    input_df = pd.DataFrame([[34., np.nan],
                             [np.nan, np.nan],
                             [np.nan, 6.],
                             [30., np.nan],
                             [13., np.nan],
                             [np.nan, np.nan],
                             [20., 7.],
                             [np.nan, np.nan],
                             [40., np.nan]], columns=["Spend", "Volume"])

    # test default parameter
    output_df = pd.DataFrame([[34., 2.],
                              [10., 2.],
                              [10., 2.],
                              [10., 1.75],
                              [13., 1.75],
                              [10., 1.75],
                              [10., 1.75],
                              [20., np.nan],
                              [20., np.nan]], columns=["Spend", "Volume"])
    pd.testing.assert_frame_equal(mmm.spread_notna(input_df), output_df)
    output_df_1 = pd.DataFrame([[11.33333333, np.nan],
                                [11.33333333, np.nan],
                                [11.33333333, 1.5],
                                [30., 1.5],
                                [6.5, 1.5],
                                [6.5, 1.5],
                                [10., 2.33333333],
                                [10., 2.33333333],
                                [40., 2.33333333]], columns=["Spend", "Volume"])
    pd.testing.assert_frame_equal(mmm.spread_notna(input_df, prior=False), output_df_1)

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
                         'Dep': [479532., 85700., 307530., 347654., 229367., 92130., 99820., 103480., 97567.5, 105322., 104699.,
                                 36128., 110005., 89533., 85296.]}, index=df_ind)
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
                         'Dep': [479532., 85700., 307530., 347654., 229367., 92130., 99820., 103480., 97567.5, 105322., 104699.,
                                 36128., 110005., 89533., 85296.]}, index=df_ind)
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


def test_assess_error():
    # Create model coefficient
    ind = [np.repeat(["CITY", "METRO"], 2),
           [("Intercept", 0, 1, 0), ("A", 0, 1, 0), ("Intercept", 0, 1, 0), ("B", 0, 1, 0)]]
    coef = pd.Series(data=[.2, .3, .4, .5], index=ind)
    # Create data
    df_ind = [np.repeat(["CITY", "METRO"], 3), np.tile(pd.date_range(start='1/1/2018', end='1/03/2018'), 2)]
    df = pd.DataFrame({'A': [773, 137, 508, 562, 365, 500], 'B': [848, 326, 969, 730, 761, 137]}, index=df_ind)
    df["Intercept"] = 1
    # Create response data
    dep = pd.DataFrame({'Dep': [773, 137, 508, 562, 365, 500]}, index=df_ind)
    # Expected Output
    dep_out = pd.DataFrame({'Dependent': [773.0, 137.0, 508.0, 562.0, 365.0, 500.0]}, index=df_ind)
    pred = pd.DataFrame({'Predicted': [232.1, 41.3, 152.6, 365.4, 380.9, 68.9]}, index=df_ind)
    error = pd.DataFrame({'Error': [540.9, 95.7, 355.4, 196.6, -15.9, 431.1]}, index=df_ind)
    error_perc = pd.DataFrame({'Error %': [0.699741, 0.69854, 0.699606, 0.349822, -0.043562, 0.8622]}, index=df_ind)
    expected_output = pd.concat([dep_out, pred, error, error_perc], axis=1)
    pd.testing.assert_frame_equal(mmm.assess_error(mmm.apply_coef(df, coef, dep_series=dep["Dep"])).round(4), expected_output.round(4))

def test_aggregate_data():
    #input data
    input_df=pd.DataFrame(np.array([[35, 96, 16, 60, 81],
           [23, 97, 79, 53, 10],
           [77, 64, 79, 67, 29],
           [16, 29, 60, 35, 16],
           [66, 52, 87, 15, 50],
           [51, 98, 84, 62, 21],
           [69, 73, 19, 37, 48]]), columns=['TV_GRP', 'TV_SPD', 'TV_IMP', 'RAD_IMP', 'RAD_SPD'])
    input_df.index=pd.MultiIndex.from_tuples([('panel1', '5/15/2020'),
                ('panel1', '5/22/2020'),
                ('panel1', '5/29/2020'),
                ('panel2', '5/15/2020'),
                ('panel2', '5/22/2020'),
                ('panel2', '5/29/2020'),
                ('panel3', '6/19/2020')],
               names=['Panel', 'Date'])
    
    # without any variable or panel aggregation
    pd.testing.assert_frame_equal(
        mmm.aggregate_data(input_df, panel_agg={}, variable_agg={}, metric_index_var=[-1], metric_mean_code=[], delimeter= "_"),
        input_df)
    
    # with panel aggregation
    output_df=pd.DataFrame(np.array([[ 51, 125,  76,  95,  97],
           [ 89, 149, 166,  68,  60],
           [128, 162, 163, 129,  50],
           [ 69,  73,  19,  37,  48]]), columns=['TV_GRP', 'TV_SPD', 'TV_IMP', 'RAD_IMP', 'RAD_SPD'])
    output_df.index=pd.MultiIndex.from_tuples([('panel5', '5/15/2020'),
                ('panel5', '5/22/2020'),
                ('panel5', '5/29/2020'),
                ('panel3', '6/19/2020')],
               names=['Panel', 'Date'])
    pd.testing.assert_frame_equal(
        mmm.aggregate_data(input_df, panel_agg={"panel5":["panel1","panel2"]}, variable_agg={}, metric_index_var=[-1], metric_mean_code=[], delimeter= "_"),
        output_df)
    
    # with panel and variable aggregation
    output_df=pd.DataFrame(np.array([[ 51, 222,  76,  95],
       [ 89, 209, 166,  68],
       [128, 212, 163, 129],
       [ 69, 121,  19,  37]]), columns=['TV_GRP', 'DIG_SPD', 'TV_IMP', 'RAD_IMP'])
    output_df.index=pd.MultiIndex.from_tuples([('panel5', '5/15/2020'),
                ('panel5', '5/22/2020'),
                ('panel5', '5/29/2020'),
                ('panel3', '6/19/2020')],
               names=['Panel', 'Date'])
    pd.testing.assert_frame_equal(
        mmm.aggregate_data(input_df, panel_agg={"panel5":["panel1","panel2"]}, variable_agg={"DIG_SPD":["TV_SPD","RAD_SPD"]}, metric_index_var=[-1], metric_mean_code=[], delimeter= "_"),
        output_df)

    # with panel and variable aggregation and mean summary
    output_df=pd.DataFrame(np.array([[ 51  , 125  ,  97  ,  42.75],
           [ 89  , 149  ,  60  ,  58.5 ],
           [128  , 162  ,  50  ,  73.  ],
           [ 69  ,  73  ,  48  ,  28.  ]]), columns=['TV_GRP', 'TV_SPD', 'RAD_SPD', 'DIG_IMP'])
    output_df.index=pd.MultiIndex.from_tuples([('panel5', '5/15/2020'),
                ('panel5', '5/22/2020'),
                ('panel5', '5/29/2020'),
                ('panel3', '6/19/2020')],
               names=['Panel', 'Date'])
    pd.testing.assert_frame_equal(
        mmm.aggregate_data(input_df, panel_agg={"panel5":["panel1","panel2"]}, variable_agg={"DIG_IMP":["TV_IMP","RAD_IMP"]}, metric_index_var=[-1], metric_mean_code=["IMP"], delimeter= "_"),
        output_df,check_dtype=False)

def test_segregate_data():
    # input data for function
    segregated_data_index=pd.MultiIndex.from_tuples([('panel1', '5/15/2020'),
                ('panel1', '5/22/2020'),
                ('panel1', '5/29/2020'),
                ('panel2', '5/15/2020'),
                ('panel2', '5/22/2020'),
                ('panel2', '5/29/2020'),
                ('panel3', '5/15/2020'),
                ('panel3', '5/22/2020'),
                ('panel3', '5/29/2020'),
                ('panel4', '5/15/2020'),
                ('panel4', '5/22/2020'),
                ('panel4', '5/29/2020'),
                ('panel5', '5/15/2020'),
                ('panel5', '5/22/2020'),
                ('panel5', '5/29/2020')],
               names=['Panel', 'Date'])
    segregated_data=pd.DataFrame([[ 46,  95,  79,  53,  94],
           [ 59,  46,  52,  52,  10],
           [ 47,  40,  87,  63,  26],
           [ 38,  12,  62,  14,  61],
           [ 71,  45,  10,  71,  34],
           [ 29,  30,  58,  44,  75],
           [ 32,  12,  91,  75, 100],
           [ 61,  89,  92,  77,  50],
           [ 46,  97,  51,  41,  63],
           [ 40,  25,  48,  16,  39],
           [ 49,  68,  53,  63,  13],
           [ 50,  69,  62,  81,  91],
           [ 12,  51,  82,  44,  35],
           [ 62,  51,  43,  63,  82],
           [ 62,  76,  19,  98,  58]],segregated_data_index, columns=['A', 'B', 'C', 'D', 'E'])
    
    # when segregated data contains extra variables
    aggregated_data_index=pd.MultiIndex.from_tuples([('panel7', '5/15/2020'),
                ('panel7', '5/22/2020'),
                ('panel7', '5/29/2020'),
                ('panel8', '5/15/2020'),
                ('panel8', '5/22/2020'),
                ('panel8', '5/29/2020'),
                ('panel5', '5/15/2020'),
                ('panel5', '5/22/2020'),
                ('panel5', '5/29/2020')],
               names=['Panel', 'Date'])
    aggregated_data=pd.DataFrame([[192, 136],
           [188, 113],
           [166, 167],
           [101, 109],
           [106, 143],
           [146, 199],
           [194, 175],
           [125, 139],
           [106, 140]],aggregated_data_index, columns=['AB', 'DE'])
    output_data_index=pd.MultiIndex.from_tuples([('panel1', '5/15/2020'),
            ('panel1', '5/22/2020'),
            ('panel1', '5/29/2020'),
            ('panel2', '5/15/2020'),
            ('panel2', '5/22/2020'),
            ('panel2', '5/29/2020'),
            ('panel3', '5/15/2020'),
            ('panel3', '5/22/2020'),
            ('panel3', '5/29/2020'),
            ('panel4', '5/15/2020'),
            ('panel4', '5/22/2020'),
            ('panel4', '5/29/2020'),
            ('panel5', '5/15/2020'),
            ('panel5', '5/22/2020'),
            ('panel5', '5/29/2020')],
           names=['Panel', 'Date'])
    output_data=pd.DataFrame([[ 46.2408377 ,  95.4973822 ,  32.46846847,  57.58558559],
           [ 50.19004525,  39.13122172,  35.18562874,   6.76646707],
           [ 53.43835616,  45.47945205,  50.58173077,  20.875     ],
           [ 38.19895288,  12.06282723,   8.57657658,  37.36936937],
           [ 60.39819005,  38.28054299,  48.04191617,  23.00598802],
           [ 32.97260274,  34.10958904,  35.32692308,  60.21634615],
           [ 29.65137615,  11.11926606,  35.54347826,  47.39130435],
           [ 24.21722846,  35.33333333,  54.24137931,  35.22167488],
           [ 25.63358779,  54.05343511,  29.5615942 ,  45.42391304],
           [ 37.06422018,  23.16513761,   7.5826087 ,  18.4826087 ],
           [ 19.45318352,  26.99625468,  44.37931034,   9.15763547],
           [ 27.86259542,  38.45038168,  58.40217391,  65.61231884],
           [ 36.95238095, 157.04761905,  97.46835443,  77.53164557],
           [ 68.5840708 ,  56.4159292 ,  60.39310345,  78.60689655],
           [ 47.62318841,  58.37681159,  87.94871795,  52.05128205]],output_data_index, columns=['A', 'B', 'D', 'E'])
    pd.testing.assert_frame_equal(mmm.segregate_data(aggregated_data, segregated_data,
                                                 panel_agg={"panel7":["panel1","panel2"],"panel8":["panel3","panel4"]},
                                                 variable_agg={"AB":["A","B"],"DE":["D","E"]}, panel_agg_index=0, match_sum=True),
                                  output_data)
    pd.testing.assert_frame_equal(mmm.segregate_data(aggregated_data, segregated_data,
                                                 panel_agg={"panel7":["panel1","panel2"],"panel8":["panel3","panel4"]},
                                                 variable_agg={"AB":["A","B"],"DE":["D","E"]}, panel_agg_index=0, match_sum=False),
                                  output_data)

def test_segregate_data():
    # input data for function
    segregated_data_index=pd.MultiIndex.from_tuples([('panel1', '5/15/2020'),
                ('panel1', '5/22/2020'),
                ('panel1', '5/29/2020'),
                ('panel2', '5/15/2020'),
                ('panel2', '5/22/2020'),
                ('panel2', '5/29/2020'),
                ('panel3', '5/15/2020'),
                ('panel3', '5/22/2020'),
                ('panel3', '5/29/2020'),
                ('panel4', '5/15/2020'),
                ('panel4', '5/22/2020'),
                ('panel4', '5/29/2020'),
                ('panel5', '5/15/2020'),
                ('panel5', '5/22/2020'),
                ('panel5', '5/29/2020')],
               names=['Panel', 'Date'])
    segregated_data=pd.DataFrame([[ 46,  95,  79,  53,  94],
           [ 59,  46,  52,  52,  10],
           [ 47,  40,  87,  63,  26],
           [ 38,  12,  62,  14,  61],
           [ 71,  45,  10,  71,  34],
           [ 29,  30,  58,  44,  75],
           [ 32,  12,  91,  75, 100],
           [ 61,  89,  92,  77,  50],
           [ 46,  97,  51,  41,  63],
           [ 40,  25,  48,  16,  39],
           [ 49,  68,  53,  63,  13],
           [ 50,  69,  62,  81,  91],
           [ 12,  51,  82,  44,  35],
           [ 62,  51,  43,  63,  82],
           [ 62,  76,  19,  98,  58]],segregated_data_index, columns=['A', 'B', 'C', 'D', 'E'])
    
    # when aggregated data contains few panels
    aggregated_data_index=pd.MultiIndex.from_tuples([('panel7', '5/15/2020'),
                ('panel7', '5/22/2020'),
                ('panel7', '5/29/2020'),
                ('panel8', '5/15/2020'),
                ('panel8', '5/22/2020'),
                ('panel8', '5/29/2020')],
               names=['Panel', 'Date'])
    aggregated_data=pd.DataFrame([[192, 197, 136],
           [188, 143, 113],
           [166, 136, 167],
           [101, 190, 109],
           [106, 123, 143],
           [146, 120, 199]],aggregated_data_index, columns=['AB', 'C', 'DE'])
    output_data_index=pd.MultiIndex.from_tuples([('panel1', '5/15/2020'),
            ('panel1', '5/22/2020'),
            ('panel1', '5/29/2020'),
            ('panel2', '5/15/2020'),
            ('panel2', '5/22/2020'),
            ('panel2', '5/29/2020'),
            ('panel3', '5/15/2020'),
            ('panel3', '5/22/2020'),
            ('panel3', '5/29/2020'),
            ('panel4', '5/15/2020'),
            ('panel4', '5/22/2020'),
            ('panel4', '5/29/2020')],
           names=['Panel', 'Date'])
    output_data=pd.DataFrame([[ 46.2408377 ,  95.4973822 ,  32.46846847,  57.58558559,
            110.37588652],
           [ 50.19004525,  39.13122172,  35.18562874,   6.76646707,
            119.93548387],
           [ 53.43835616,  45.47945205,  50.58173077,  20.875     ,
             81.6       ],
           [ 38.19895288,  12.06282723,   8.57657658,  37.36936937,
             86.62411348],
           [ 60.39819005,  38.28054299,  48.04191617,  23.00598802,
             23.06451613],
           [ 32.97260274,  34.10958904,  35.32692308,  60.21634615,
             54.4       ],
           [ 29.65137615,  11.11926606,  35.54347826,  47.39130435,
            124.38848921],
           [ 24.21722846,  35.33333333,  54.24137931,  35.22167488,
             78.04137931],
           [ 25.63358779,  54.05343511,  29.5615942 ,  45.42391304,
             54.15929204],
           [ 37.06422018,  23.16513761,   7.5826087 ,  18.4826087 ,
             65.61151079],
           [ 19.45318352,  26.99625468,  44.37931034,   9.15763547,
             44.95862069],
           [ 27.86259542,  38.45038168,  58.40217391,  65.61231884,
             65.84070796]],output_data_index, columns=['A', 'B', 'D', 'E', 'C'])
    pd.testing.assert_frame_equal(mmm.segregate_data(aggregated_data, segregated_data,
                                                 panel_agg={"panel7":["panel1","panel2"],"panel8":["panel3","panel4"]},
                                                 variable_agg={"AB":["A","B"],"DE":["D","E"]}, panel_agg_index=0, match_sum=True),
                                  output_data)
    pd.testing.assert_frame_equal(mmm.segregate_data(aggregated_data, segregated_data,
                                                 panel_agg={"panel7":["panel1","panel2"],"panel8":["panel3","panel4"]},
                                                 variable_agg={"AB":["A","B"],"DE":["D","E"]}, panel_agg_index=0, match_sum=False),
                                  output_data)
                                  
    # combination of aggregated panel and non aggregated panel in aggregated data
    aggregated_data_index=pd.MultiIndex.from_tuples([('panel7', '5/15/2020'),
                ('panel7', '5/22/2020'),
                ('panel7', '5/29/2020'),
                ('panel8', '5/15/2020'),
                ('panel8', '5/22/2020'),
                ('panel8', '5/29/2020'),
                ('panel5', '5/15/2020'),
                ('panel5', '5/22/2020'),
                ('panel5', '5/29/2020')],
               names=['Panel', 'Date'])
    aggregated_data=pd.DataFrame([[192, 197, 136],
           [188, 143, 113],
           [166, 136, 167],
           [101, 190, 109],
           [106, 123, 143],
           [146, 120, 199],
           [194, 175, 175],
           [125, 194, 139],
           [106, 125, 140]],aggregated_data_index, columns=['AB', 'C', 'DE'])
    output_data_index=pd.MultiIndex.from_tuples([('panel1', '5/15/2020'),
            ('panel1', '5/22/2020'),
            ('panel1', '5/29/2020'),
            ('panel2', '5/15/2020'),
            ('panel2', '5/22/2020'),
            ('panel2', '5/29/2020'),
            ('panel3', '5/15/2020'),
            ('panel3', '5/22/2020'),
            ('panel3', '5/29/2020'),
            ('panel4', '5/15/2020'),
            ('panel4', '5/22/2020'),
            ('panel4', '5/29/2020'),
            ('panel5', '5/15/2020'),
            ('panel5', '5/22/2020'),
            ('panel5', '5/29/2020')],
           names=['Panel', 'Date'])
    output_data=pd.DataFrame([[ 46.2408377 ,  95.4973822 ,  32.46846847,  57.58558559,
        110.37588652],
       [ 50.19004525,  39.13122172,  35.18562874,   6.76646707,
        119.93548387],
       [ 53.43835616,  45.47945205,  50.58173077,  20.875     ,
         81.6       ],
       [ 38.19895288,  12.06282723,   8.57657658,  37.36936937,
         86.62411348],
       [ 60.39819005,  38.28054299,  48.04191617,  23.00598802,
         23.06451613],
       [ 32.97260274,  34.10958904,  35.32692308,  60.21634615,
         54.4       ],
       [ 29.65137615,  11.11926606,  35.54347826,  47.39130435,
        124.38848921],
       [ 24.21722846,  35.33333333,  54.24137931,  35.22167488,
         78.04137931],
       [ 25.63358779,  54.05343511,  29.5615942 ,  45.42391304,
         54.15929204],
       [ 37.06422018,  23.16513761,   7.5826087 ,  18.4826087 ,
         65.61151079],
       [ 19.45318352,  26.99625468,  44.37931034,   9.15763547,
         44.95862069],
       [ 27.86259542,  38.45038168,  58.40217391,  65.61231884,
         65.84070796],
       [ 36.95238095, 157.04761905,  97.46835443,  77.53164557,
        175.        ],
       [ 68.5840708 ,  56.4159292 ,  60.39310345,  78.60689655,
        194.        ],
       [ 47.62318841,  58.37681159,  87.94871795,  52.05128205,
        125.        ]],output_data_index, columns=['A', 'B', 'D', 'E', 'C'])
    pd.testing.assert_frame_equal(mmm.segregate_data(aggregated_data, segregated_data,
                                                 panel_agg={"panel7":["panel1","panel2"],"panel8":["panel3","panel4"]},
                                                 variable_agg={"AB":["A","B"],"DE":["D","E"]}, panel_agg_index=0, match_sum=True),
                                  output_data)
    pd.testing.assert_frame_equal(mmm.segregate_data(aggregated_data, segregated_data,
                                                 panel_agg={"panel7":["panel1","panel2"],"panel8":["panel3","panel4"]},
                                                 variable_agg={"AB":["A","B"],"DE":["D","E"]}, panel_agg_index=0, match_sum=False),
                                  output_data)

    #Cases when segregated data have 0 values
    segregated_data_index=pd.MultiIndex.from_tuples([('panel1', '5/15/2020'),
                ('panel1', '5/22/2020'),
                ('panel1', '5/29/2020'),
                ('panel2', '5/15/2020'),
                ('panel2', '5/22/2020'),
                ('panel2', '5/29/2020'),
                ('panel3', '5/15/2020'),
                ('panel3', '5/22/2020'),
                ('panel3', '5/29/2020'),
                ('panel4', '5/15/2020'),
                ('panel4', '5/22/2020'),
                ('panel4', '5/29/2020'),
                ('panel5', '5/15/2020'),
                ('panel5', '5/22/2020'),
                ('panel5', '5/29/2020')],
               names=['Panel', 'Date'])
    segregated_data=pd.DataFrame([[ 0,  0,  79,  53,  94],
           [ 59,  46,  0,  52,  10],
           [ 47,  40,  87,  63,  26],
           [ 0,  0,  62,  14,  61],
           [ 0,  0,  0,  71,  34],
           [ 29,  30,  58,  44,  75],
           [ 32,  12,  91,  0, 0],
           [ 61,  89,  92,  77,  50],
           [ 46,  97,  51,  41,  63],
           [ 40,  25,  0,  16,  39],
           [ 49,  68,  53,  63,  13],
           [ 50,  69,  62,  81,  91],
           [ 12,  51,  82,  44,  35],
           [ 62,  51,  0,  63,  82],
           [ 62,  76,  0,  0,  0]],segregated_data_index, columns=['A', 'B', 'C', 'D', 'E'])
    output_data_index=pd.MultiIndex.from_tuples([('panel1', '5/15/2020'),
            ('panel1', '5/22/2020'),
            ('panel1', '5/29/2020'),
            ('panel2', '5/15/2020'),
            ('panel2', '5/22/2020'),
            ('panel2', '5/29/2020'),
            ('panel3', '5/15/2020'),
            ('panel3', '5/22/2020'),
            ('panel3', '5/29/2020'),
            ('panel4', '5/15/2020'),
            ('panel4', '5/22/2020'),
            ('panel4', '5/29/2020'),
            ('panel5', '5/15/2020'),
            ('panel5', '5/22/2020'),
            ('panel5', '5/29/2020')],
           names=['Panel', 'Date'])
    output_data=pd.DataFrame([[         np.nan,          np.nan,  36.31739784,  64.41198862,
            157.7745405 ],
           [123.55551068,  96.33141511,  39.35666009,   7.56858848,
                     np.nan],
           [ 62.50210562,  53.19328138,  56.57787159,  23.3495978 ,
            116.64144144],
           [         np.nan,          np.nan,   9.5932749 ,  41.79926921,
            123.8230571 ],
           [  0.        ,   0.        ,  53.7369782 ,  25.73320083,
                     np.nan],
           [ 38.565129  ,  39.89496103,  39.51470397,  67.35460903,
             77.76096096],
           [ 34.68058482,  13.00521931,   0.        ,   0.        ,
            190.        ],
           [ 28.32474425,  41.3262662 ,  60.67134807,  39.39697927,
             78.04137931],
           [ 29.98133412,  63.22150891,  33.06593221,  50.80862754,
             54.15929204],
           [ 43.35073103,  27.09420689,  35.46800092,  86.45325225,
              0.        ],
           [ 22.75266341,  31.57512473,  49.64019387,  10.24321461,
             44.95862069],
           [ 32.58840666,  44.97200119,  65.32537827,  73.39023979,
             65.84070796],
           [ 43.2199226 , 183.68467104, 109.02260474,  86.7225265 ,
            175.        ],
           [ 80.21670471,  65.98470872,  67.55231978,  87.92524161,
            194.        ],
           [ 55.70061965,  68.27817893,          np.nan,          np.nan,
            125.        ]],output_data_index, columns=['A', 'B', 'D', 'E', 'C'])
    pd.testing.assert_frame_equal(mmm.segregate_data(aggregated_data, segregated_data,
                                                 panel_agg={"panel7":["panel1","panel2"],"panel8":["panel3","panel4"]},
                                                 variable_agg={"AB":["A","B"],"DE":["D","E"]}, panel_agg_index=0, match_sum=True),
                                  output_data)
    # case when match_sum=False
    output_data_index=pd.MultiIndex.from_tuples([('panel1', '5/15/2020'),
            ('panel1', '5/22/2020'),
            ('panel1', '5/29/2020'),
            ('panel2', '5/15/2020'),
            ('panel2', '5/22/2020'),
            ('panel2', '5/29/2020'),
            ('panel3', '5/15/2020'),
            ('panel3', '5/22/2020'),
            ('panel3', '5/29/2020'),
            ('panel4', '5/15/2020'),
            ('panel4', '5/22/2020'),
            ('panel4', '5/29/2020'),
            ('panel5', '5/15/2020'),
            ('panel5', '5/22/2020'),
            ('panel5', '5/29/2020')],
           names=['Panel', 'Date'])
    output_data=pd.DataFrame([[         np.nan,          np.nan,  32.46846847,  57.58558559,
            110.37588652],
           [105.63809524,  82.36190476,  35.18562874,   6.76646707,
                     np.nan],
           [ 53.43835616,  45.47945205,  50.58173077,  20.875     ,
             81.6       ],
           [         np.nan,          np.nan,   8.57657658,  37.36936937,
             86.62411348],
           [  0.        ,   0.        ,  48.04191617,  23.00598802,
                     np.nan],
           [ 32.97260274,  34.10958904,  35.32692308,  60.21634615,
             54.4       ],
           [ 29.65137615,  11.11926606,   0.        ,   0.        ,
            190.        ],
           [ 24.21722846,  35.33333333,  54.24137931,  35.22167488,
             78.04137931],
           [ 25.63358779,  54.05343511,  29.5615942 ,  45.42391304,
             54.15929204],
           [ 37.06422018,  23.16513761,  31.70909091,  77.29090909,
              0.        ],
           [ 19.45318352,  26.99625468,  44.37931034,   9.15763547,
             44.95862069],
           [ 27.86259542,  38.45038168,  58.40217391,  65.61231884,
             65.84070796],
           [ 36.95238095, 157.04761905,  97.46835443,  77.53164557,
            175.        ],
           [ 68.5840708 ,  56.4159292 ,  60.39310345,  78.60689655,
            194.        ],
           [ 47.62318841,  58.37681159,          np.nan,          np.nan,
            125.        ]],output_data_index, columns=['A', 'B', 'D', 'E', 'C'])
    pd.testing.assert_frame_equal(mmm.segregate_data(aggregated_data, segregated_data,
                                                 panel_agg={"panel7":["panel1","panel2"],"panel8":["panel3","panel4"]},
                                                 variable_agg={"AB":["A","B"],"DE":["D","E"]}, panel_agg_index=0, match_sum=False),
                                  output_data)

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
                         'D': [479532., 85700., 307530., 347654., 229367., 92130., 99820., 103480., 97567.5, 105322., 104699., 36128.,
                               110005., 89533., 85296.]}, index=df_ind)
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
    pd.testing.assert_frame_equal(mmm.apply_coef(df_2, coef_2, dep_series=None), expected_output)
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
                         'D': [479532., 85700., 307530., 347654., 229367., 92130., 99820., 103480., 97567.5, 105322., 104699., 36128.,
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
    pd.testing.assert_frame_equal(mmm.apply_coef(df_2, coef_2, dep_series=dep["Dep"]), expected_output)

    # different node in many panels
    df_2={'A': {('Panel1', '5/15/2020'): 150,
      ('Panel1', '5/22/2020'): 115,
      ('Panel1', '5/29/2020'): 166,
      ('Panel1', '6/5/2020'): 171,
      ('Panel1', '6/12/2020'): 195,
      ('Panel1', '6/19/2020'): 150,
      ('Panel1', '6/26/2020'): 127,
      ('Panel2', '5/15/2020'): 150,
      ('Panel2', '5/22/2020'): 115,
      ('Panel2', '5/29/2020'): 166,
      ('Panel2', '6/5/2020'): 171,
      ('Panel2', '6/12/2020'): 195,
      ('Panel2', '6/19/2020'): 150,
      ('Panel2', '6/26/2020'): 127},
     'B': {('Panel1', '5/15/2020'): 196,
      ('Panel1', '5/22/2020'): 150,
      ('Panel1', '5/29/2020'): 111,
      ('Panel1', '6/5/2020'): 148,
      ('Panel1', '6/12/2020'): 171,
      ('Panel1', '6/19/2020'): 179,
      ('Panel1', '6/26/2020'): 142,
      ('Panel2', '5/15/2020'): 196,
      ('Panel2', '5/22/2020'): 150,
      ('Panel2', '5/29/2020'): 111,
      ('Panel2', '6/5/2020'): 148,
      ('Panel2', '6/12/2020'): 171,
      ('Panel2', '6/19/2020'): 179,
      ('Panel2', '6/26/2020'): 142},
     'C': {('Panel1', '5/15/2020'): 148,
      ('Panel1', '5/22/2020'): 182,
      ('Panel1', '5/29/2020'): 162,
      ('Panel1', '6/5/2020'): 115,
      ('Panel1', '6/12/2020'): 171,
      ('Panel1', '6/19/2020'): 165,
      ('Panel1', '6/26/2020'): 189,
      ('Panel2', '5/15/2020'): 148,
      ('Panel2', '5/22/2020'): 182,
      ('Panel2', '5/29/2020'): 162,
      ('Panel2', '6/5/2020'): 115,
      ('Panel2', '6/12/2020'): 171,
      ('Panel2', '6/19/2020'): 165,
      ('Panel2', '6/26/2020'): 189},
     'D': {('Panel1', '5/15/2020'): 171,
      ('Panel1', '5/22/2020'): 172,
      ('Panel1', '5/29/2020'): 141,
      ('Panel1', '6/5/2020'): 153,
      ('Panel1', '6/12/2020'): 150,
      ('Panel1', '6/19/2020'): 101,
      ('Panel1', '6/26/2020'): 149,
      ('Panel2', '5/15/2020'): 171,
      ('Panel2', '5/22/2020'): 172,
      ('Panel2', '5/29/2020'): 141,
      ('Panel2', '6/5/2020'): 153,
      ('Panel2', '6/12/2020'): 150,
      ('Panel2', '6/19/2020'): 101,
      ('Panel2', '6/26/2020'): 149},
     'Dep': {('Panel1', '5/15/2020'): 673,
      ('Panel1', '5/22/2020'): 588,
      ('Panel1', '5/29/2020'): 553,
      ('Panel1', '6/5/2020'): 690,
      ('Panel1', '6/12/2020'): 785,
      ('Panel1', '6/19/2020'): 699,
      ('Panel1', '6/26/2020'): 569,
      ('Panel2', '5/15/2020'): 673,
      ('Panel2', '5/22/2020'): 588,
      ('Panel2', '5/29/2020'): 553,
      ('Panel2', '6/5/2020'): 690,
      ('Panel2', '6/12/2020'): 785,
      ('Panel2', '6/19/2020'): 699,
      ('Panel2', '6/26/2020'): 569},
     'Intercept': {('Panel1', '5/15/2020'): 1,
      ('Panel1', '5/22/2020'): 1,
      ('Panel1', '5/29/2020'): 1,
      ('Panel1', '6/5/2020'): 1,
      ('Panel1', '6/12/2020'): 1,
      ('Panel1', '6/19/2020'): 1,
      ('Panel1', '6/26/2020'): 1,
      ('Panel2', '5/15/2020'): 1,
      ('Panel2', '5/22/2020'): 1,
      ('Panel2', '5/29/2020'): 1,
      ('Panel2', '6/5/2020'): 1,
      ('Panel2', '6/12/2020'): 1,
      ('Panel2', '6/19/2020'): 1,
      ('Panel2', '6/26/2020'): 1},
     'F': {('Panel1', '5/15/2020'): 119,
      ('Panel1', '5/22/2020'): 165,
      ('Panel1', '5/29/2020'): 115,
      ('Panel1', '6/5/2020'): 139,
      ('Panel1', '6/12/2020'): 112,
      ('Panel1', '6/19/2020'): 116,
      ('Panel1', '6/26/2020'): 176,
      ('Panel2', '5/15/2020'): 119,
      ('Panel2', '5/22/2020'): 165,
      ('Panel2', '5/29/2020'): 115,
      ('Panel2', '6/5/2020'): 139,
      ('Panel2', '6/12/2020'): 112,
      ('Panel2', '6/19/2020'): 116,
      ('Panel2', '6/26/2020'): 176}}
    df_2=pd.DataFrame(df_2, columns=['A', 'B', 'C', 'D', 'Dep', 'Intercept', 'F'])
    df_2.index.names=["Panel","Date"]
    coef_input=pd.Series([3.24e+03, 6.00e+02, 1.00e+01, 9.00e-01, 3.24e+03, 1.00e+01,
       9.00e-01, 3.24e+03, 6.00e+02, 1.00e+01, 9.00e-01, 3.24e+03,
       1.00e+01, 9.00e-01],
         index=pd.MultiIndex.from_tuples([('Panel1', ('Intercept', 0, 1, 0)),
            ('Panel1',         ('A', 0, 1, 0)),
            ('Panel1',         ('B', 0, 1, 0)),
            ('Panel1',         ('C', 0, 1, 0)),
            ('Panel1', ('Intercept', 0, 1, 0)),
            ('Panel1',         ('B', 0, 1, 0)),
            ('Panel1',         ('D', 0, 1, 0)),
            ('Panel2', ('Intercept', 0, 1, 0)),
            ('Panel2',         ('A', 0, 1, 0)),
            ('Panel2',         ('B', 0, 1, 0)),
            ('Panel2',         ('C', 0, 1, 0)),
            ('Panel2', ('Intercept', 0, 1, 0)),
            ('Panel2',         ('B', 0, 1, 0)),
            ('Panel2',         ('D', 0, 1, 0))],
           names=['Panel', 'From']))
    node_input=pd.Series([('Dep', 0, 1, 0), ('Dep', 0, 1, 0), ('Dep', 0, 1, 0),
       ('Dep', 0, 1, 0), ('A', 0, 1, 0), ('A', 0, 1, 0), ('A', 0, 1, 0),
       ('Dep', 0, 1, 0), ('Dep', 0, 1, 0), ('Dep', 0, 1, 0),
       ('Dep', 0, 1, 0), ('C', 0, 1, 0), ('C', 0, 1, 0), ('C', 0, 1, 0)],
         index=pd.MultiIndex.from_tuples([('Panel1', ('Intercept', 0, 1, 0)),
            ('Panel1',         ('A', 0, 1, 0)),
            ('Panel1',         ('B', 0, 1, 0)),
            ('Panel1',         ('C', 0, 1, 0)),
            ('Panel1', ('Intercept', 0, 1, 0)),
            ('Panel1',         ('B', 0, 1, 0)),
            ('Panel1',         ('D', 0, 1, 0)),
            ('Panel2', ('Intercept', 0, 1, 0)),
            ('Panel2',         ('A', 0, 1, 0)),
            ('Panel2',         ('B', 0, 1, 0)),
            ('Panel2',         ('C', 0, 1, 0)),
            ('Panel2', ('Intercept', 0, 1, 0)),
            ('Panel2',         ('B', 0, 1, 0)),
            ('Panel2',         ('D', 0, 1, 0))],
           names=['Panel', 'From']))
    output=pd.DataFrame([[-1.963900e+03,  1.960000e+03,  1.539000e+02, -1.500000e+02,
                      np.nan,           np.nan,           np.nan,           np.nan,
             9.000000e+04,  1.960000e+03,  1.332000e+02,  3.240000e+03,
            -9.466020e+04],
           [-1.539800e+03,  1.500000e+03,  1.548000e+02, -1.150000e+02,
                      np.nan,           np.nan,           np.nan,           np.nan,
             6.900000e+04,  1.500000e+03,  1.638000e+02,  3.240000e+03,
            -7.331580e+04],
           [-1.070900e+03,  1.110000e+03,  1.269000e+02, -1.660000e+02,
                      np.nan,           np.nan,           np.nan,           np.nan,
             9.960000e+04,  1.110000e+03,  1.458000e+02,  3.240000e+03,
            -1.035428e+05],
           [-1.446700e+03,  1.480000e+03,  1.377000e+02, -1.710000e+02,
                      np.nan,           np.nan,           np.nan,           np.nan,
             1.026000e+05,  1.480000e+03,  1.035000e+02,  3.240000e+03,
            -1.067335e+05],
           [-1.650000e+03,  1.710000e+03,  1.350000e+02, -1.950000e+02,
                      np.nan,           np.nan,           np.nan,           np.nan,
             1.170000e+05,  1.710000e+03,  1.539000e+02,  3.240000e+03,
            -1.213189e+05],
           [-1.730900e+03,  1.790000e+03,  9.090000e+01, -1.500000e+02,
                      np.nan,           np.nan,           np.nan,           np.nan,
             9.000000e+04,  1.790000e+03,  1.485000e+02,  3.240000e+03,
            -9.447950e+04],
           [-1.427100e+03,  1.420000e+03,  1.341000e+02, -1.270000e+02,
                      np.nan,           np.nan,           np.nan,           np.nan,
             7.620000e+04,  1.420000e+03,  1.701000e+02,  3.240000e+03,
            -8.046110e+04],
           [          np.nan,           np.nan,           np.nan,           np.nan,
             1.960000e+03, -1.965900e+03,  1.539000e+02, -1.480000e+02,
             9.000000e+04,  1.960000e+03,  1.332000e+02,  3.240000e+03,
            -9.466020e+04],
           [          np.nan,           np.nan,           np.nan,           np.nan,
             1.500000e+03, -1.472800e+03,  1.548000e+02, -1.820000e+02,
             6.900000e+04,  1.500000e+03,  1.638000e+02,  3.240000e+03,
            -7.331580e+04],
           [          np.nan,           np.nan,           np.nan,           np.nan,
             1.110000e+03, -1.074900e+03,  1.269000e+02, -1.620000e+02,
             9.960000e+04,  1.110000e+03,  1.458000e+02,  3.240000e+03,
            -1.035428e+05],
           [          np.nan,           np.nan,           np.nan,           np.nan,
             1.480000e+03, -1.502700e+03,  1.377000e+02, -1.150000e+02,
             1.026000e+05,  1.480000e+03,  1.035000e+02,  3.240000e+03,
            -1.067335e+05],
           [          np.nan,           np.nan,           np.nan,           np.nan,
             1.710000e+03, -1.674000e+03,  1.350000e+02, -1.710000e+02,
             1.170000e+05,  1.710000e+03,  1.539000e+02,  3.240000e+03,
            -1.213189e+05],
           [          np.nan,           np.nan,           np.nan,           np.nan,
             1.790000e+03, -1.715900e+03,  9.090000e+01, -1.650000e+02,
             9.000000e+04,  1.790000e+03,  1.485000e+02,  3.240000e+03,
            -9.447950e+04],
           [          np.nan,           np.nan,           np.nan,           np.nan,
             1.420000e+03, -1.365100e+03,  1.341000e+02, -1.890000e+02,
             7.620000e+04,  1.420000e+03,  1.701000e+02,  3.240000e+03,
            -8.046110e+04]],
                 index=pd.MultiIndex.from_tuples([('Panel1', '5/15/2020'),
                ('Panel1', '5/22/2020'),
                ('Panel1', '5/29/2020'),
                ('Panel1',  '6/5/2020'),
                ('Panel1', '6/12/2020'),
                ('Panel1', '6/19/2020'),
                ('Panel1', '6/26/2020'),
                ('Panel2', '5/15/2020'),
                ('Panel2', '5/22/2020'),
                ('Panel2', '5/29/2020'),
                ('Panel2',  '6/5/2020'),
                ('Panel2', '6/12/2020'),
                ('Panel2', '6/19/2020'),
                ('Panel2', '6/26/2020')],
               names=['Panel', 'Date']),
                 columns=pd.MultiIndex.from_tuples([(  ('A', 0, 1, 0), 'A',  0,  1,  0),
                (  ('A', 0, 1, 0),                   'B',  0,  1,  0),
                (  ('A', 0, 1, 0),                   'D',  0,  1,  0),
                (  ('A', 0, 1, 0), '_Adjustment_Factor_', '', '', ''),
                (  ('C', 0, 1, 0),                   'B',  0,  1,  0),
                (  ('C', 0, 1, 0),                   'C',  0,  1,  0),
                (  ('C', 0, 1, 0),                   'D',  0,  1,  0),
                (  ('C', 0, 1, 0), '_Adjustment_Factor_', '', '', ''),
                (('Dep', 0, 1, 0),                   'A',  0,  1,  0),
                (('Dep', 0, 1, 0),                   'B',  0,  1,  0),
                (('Dep', 0, 1, 0),                   'C',  0,  1,  0),
                (('Dep', 0, 1, 0),           'Intercept',  0,  1,  0),
                (('Dep', 0, 1, 0),            'Residual',  0,  1,  0)],
               names=[None, 'Variable', 'Adstock', 'Power', 'Lag']))
    pd.testing.assert_frame_equal(mmm.apply_coef(df_2, coef_input, nodes=node_input), output)