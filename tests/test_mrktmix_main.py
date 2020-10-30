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
    pd.testing.assert_frame_equal(mmm.assess_error(mmm.apply_coef(df, coef, dep["Dep"])).round(4), expected_output.round(4))
