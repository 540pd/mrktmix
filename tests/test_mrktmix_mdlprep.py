import random

import numpy as np
import pandas as pd

from mrktmix.dataprep import mdl_data as raw_dp


def test_generate_code():
    # test default value
    random.seed(999)
    assert raw_dp.generate_code(3) == '2C5'
    # test with given code length
    random.seed(999)
    assert raw_dp.generate_code(5) == '2C5UR'
    # test with given chars
    random.seed(999)
    assert raw_dp.generate_code(5, chars="ABC") == 'CACBB'


def test_apply_mapping():
    # Case sensitive = True feature
    long_format_data = {'Panel': {0: 'panel1', 1: 'panel1', 2: 'panel1', 3: 'panel1'},
                        'Channel': {0: 'TV_Free', 1: 'TV_free', 2: 'TV', 3: 'TV'},
                        'Metric': {0: 'GRp', 1: 'Spend', 2: 'GRP', 3: 'GRP'}}
    long_format_data = pd.DataFrame.from_dict(long_format_data)
    output_data = {
        'Panel': {0: 'panel1', 1: 'panel1', 2: 'panel1', 3: 'panel1'},
        'Channel': {0: 'TV_Free', 1: 'TV_free', 2: 'TV', 3: 'TV'},
        'Metric': {0: 'GRp', 1: 'Spend', 2: 'GRP', 3: 'GRP'},
        '_Variable_': {0: 'EVE_RRG', 1: 'ETE_SPE', 2: 'TVT_PGP', 3: 'TVT_PGP'}}
    output_data = pd.DataFrame.from_dict(output_data)
    output_dict = {'Spend': 'SPE', 'TV': 'TVT', 'GRP': 'PGP', 'GRp': 'RRG', 'TV_Free': 'EVE', 'TV_free': 'ETE'}
    random.seed(999)
    mapped_data, description_mapping = raw_dp.apply_mapping(
        long_format_data, ['Channel', 'Metric'], delimeter="_", description2code={}, case_sensitive=True)
    pd.testing.assert_frame_equal(mapped_data, output_data)
    assert output_dict == description_mapping

    # Case sensitive = False feature
    long_format_data = {'Panel': {0: 'panel1', 1: 'panel1', 2: 'panel1', 3: 'panel1'},
                        'Channel': {0: 'TV_Free', 1: 'TV_free', 2: 'TV', 3: 'TV'},
                        'Metric': {0: 'GRp', 1: 'Spend', 2: 'GRP', 3: 'GRP'}}
    long_format_data = pd.DataFrame.from_dict(long_format_data)
    output_data = {
        'Panel': {0: 'panel1', 1: 'panel1', 2: 'panel1', 3: 'panel1'},
        'Channel': {0: 'TV_Free', 1: 'TV_free', 2: 'TV', 3: 'TV'},
        'Metric': {0: 'GRp', 1: 'Spend', 2: 'GRP', 3: 'GRP'},
        '_Variable_': {0: 'EVE_PGP', 1: 'EVE_SPE', 2: 'TVT_PGP', 3: 'TVT_PGP'}}
    output_data = pd.DataFrame.from_dict(output_data)
    output_dict = {'Spend': 'SPE', 'TV': 'TVT', 'GRP': 'PGP', 'GRp': 'PGP', 'TV_Free': 'EVE', 'TV_free': 'EVE'}
    random.seed(999)
    mapped_data, description_mapping = raw_dp.apply_mapping(
        long_format_data, ['Channel', 'Metric'], delimeter="_", description2code={}, case_sensitive=False)
    pd.testing.assert_frame_equal(mapped_data, output_data)
    assert output_dict == description_mapping

    # Test different delimeter
    long_format_data = {'Panel': {0: 'panel1', 1: 'panel1', 2: 'panel1', 3: 'panel1'},
                        'Channel': {0: 'TV_Free', 1: 'TV_free', 2: 'TV', 3: 'TV'},
                        'Metric': {0: 'GRp', 1: 'Spend', 2: 'GRP', 3: 'GRP'}}
    long_format_data = pd.DataFrame.from_dict(long_format_data)
    output_data = {
        'Panel': {0: 'panel1', 1: 'panel1', 2: 'panel1', 3: 'panel1'},
        'Channel': {0: 'TV_Free', 1: 'TV_free', 2: 'TV', 3: 'TV'},
        'Metric': {0: 'GRp', 1: 'Spend', 2: 'GRP', 3: 'GRP'},
        '_Variable_': {0: 'EVE.PGP', 1: 'EVE.SPE', 2: 'TVT.PGP', 3: 'TVT.PGP'}}
    output_data = pd.DataFrame.from_dict(output_data)
    output_dict = {'Spend': 'SPE', 'TV': 'TVT', 'GRP': 'PGP', 'GRp': 'PGP', 'TV_Free': 'EVE', 'TV_free': 'EVE'}
    random.seed(999)
    mapped_data, description_mapping = raw_dp.apply_mapping(
        long_format_data, ['Channel', 'Metric'], delimeter=".", description2code={}, case_sensitive=False)
    pd.testing.assert_frame_equal(mapped_data, output_data)
    assert output_dict == description_mapping

    # Test predefined mapping feature when case sensitive is true
    long_format_data = {'Panel': {0: 'panel1', 1: 'panel1', 2: 'panel1', 3: 'panel1'},
                        'Channel': {0: 'TV_Free', 1: 'TV_free', 2: 'TV', 3: 'TV'},
                        'Metric': {0: 'GRp', 1: 'Spend', 2: 'GRP', 3: 'GRP'}}
    long_format_data = pd.DataFrame.from_dict(long_format_data)
    output_data = {
        'Panel': {0: 'panel1', 1: 'panel1', 2: 'panel1', 3: 'panel1'},
        'Channel': {0: 'TV_Free', 1: 'TV_free', 2: 'TV', 3: 'TV'},
        'Metric': {0: 'GRp', 1: 'Spend', 2: 'GRP', 3: 'GRP'},
        '_Variable_': {0: 'ETE.GRP', 1: 'RFT.SPE', 2: 'tv.TV_GRP', 3: 'tv.TV_GRP'}}
    output_data = pd.DataFrame.from_dict(output_data)
    output_dict = {'GRP': 'TV_GRP', 'TV': 'tv', 'GRp': 'GRP', 'Spend': 'SPE', 'TV_Free': 'ETE', 'TV_free': 'RFT'}
    random.seed(999)
    mapped_data, description_mapping = raw_dp.apply_mapping(
        long_format_data, [
            'Channel', 'Metric'], delimeter=".", description2code={
            'GRP': 'TV_GRP', "TV": "tv"}, case_sensitive=True)
    pd.testing.assert_frame_equal(mapped_data, output_data)
    assert output_dict == description_mapping

    # Test predefined mapping feature when case sensitive is false
    long_format_data = {'Panel': {0: 'panel1', 1: 'panel1', 2: 'panel1', 3: 'panel1'},
                        'Channel': {0: 'TV_Free', 1: 'TV_free', 2: 'TV', 3: 'TV'},
                        'Metric': {0: 'GRp', 1: 'Spend', 2: 'GRP', 3: 'GRP'}}
    long_format_data = pd.DataFrame.from_dict(long_format_data)
    output_data = {
        'Panel': {0: 'panel1', 1: 'panel1', 2: 'panel1', 3: 'panel1'},
        'Channel': {0: 'TV_Free', 1: 'TV_free', 2: 'TV', 3: 'TV'},
        'Metric': {0: 'GRp', 1: 'Spend', 2: 'GRP', 3: 'GRP'},
        '_Variable_': {0: 'ETE.TV_GRP', 1: 'ETE.SPE', 2: 'tv.TV_GRP', 3: 'tv.TV_GRP'}}
    output_data = pd.DataFrame.from_dict(output_data)
    output_dict = {'GRP': 'TV_GRP', 'GRp': 'TV_GRP', 'TV': 'tv', 'Spend': 'SPE', 'TV_Free': 'ETE', 'TV_free': 'ETE'}
    random.seed(999)
    mapped_data, description_mapping = raw_dp.apply_mapping(long_format_data, ['Channel', 'Metric'], delimeter=".", description2code={
                                                            'GRP': 'TV_GRP', "TV": "tv", "TV": "tv"}, case_sensitive=False)
    pd.testing.assert_frame_equal(mapped_data, output_data)
    assert output_dict == description_mapping


def test_long2wide():
    # input data
    input_data = {'Panel': {0: 'panel1',
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
                  'Metric_Value': {0: 33, 1: 102, 2: 45, 3: 129, 4: 170, 5: 24, 6: 49},
                  'Variable': {0: 'GRP',
                               1: 'SPD',
                               2: 'GRP',
                               3: 'SPD',
                               4: 'SPD',
                               5: 'GRP',
                               6: 'GRP'}}
    input_data = pd.DataFrame.from_dict(input_data)

    # Test when summary type is case sensitive
    output = {('mean', 'GRP'): {'panel1': 33.0, 'panel2': np.nan, 'panel3': np.nan},
              ('sum', 'GRP'): {'panel1': np.nan, 'panel2': 45.0, 'panel3': 73.0},
              ('sum', 'SPD'): {'panel1': 102.0, 'panel2': 299.0, 'panel3': np.nan}}
    output = pd.DataFrame.from_dict(output)
    output.columns.names = ["_summary_type_", 'Variable']
    output.index.names = ['Panel']
    pd.testing.assert_frame_equal(raw_dp.long2wide(input_data, ["Panel"], "Metric", 'Metric_Value', {
                                  'mean': ['GRp']}, variable_name='Variable', case_sensitive=True), output)

    # Test when summary type is not case sensitive
    output = {('mean', 'GRP'): {'panel1': 33.0, 'panel2': 45.0, 'panel3': 36.5},
              ('sum', 'SPD'): {'panel1': 102.0, 'panel2': 299.0, 'panel3': np.nan}}
    output = pd.DataFrame.from_dict(output)
    output.columns.names = ["_summary_type_", 'Variable']
    output.index.names = ['Panel']
    pd.testing.assert_frame_equal(raw_dp.long2wide(input_data, ["Panel"], "Metric", 'Metric_Value', {
                                  'mean': ['GRp']}, variable_name='Variable', case_sensitive=False), output)

    # multiple summary type for given metric
    output = {('mean', 'GRP'): {'panel1': 33.0, 'panel2': 45.0, 'panel3': 36.5},
              ('max', 'GRP'): {'panel1': 33, 'panel2': 45, 'panel3': 49},
              ('sum', 'SPD'): {'panel1': 102.0, 'panel2': 299.0, 'panel3': np.nan}}
    output = pd.DataFrame.from_dict(output)
    output.columns.names = ["_summary_type_", 'Variable']
    output.index.names = ['Panel']
    pd.testing.assert_frame_equal(raw_dp.long2wide(input_data, ["Panel"], "Metric", 'Metric_Value', {
                                  'mean': ['Grp'], 'max': ['Grp']}, variable_name='Variable', case_sensitive=False), output)

    # multiple summary type for given metric
    output = {('mean', 'GRP'): {'panel1': 33.0, 'panel2': 45.0, 'panel3': 36.5},
              ('mean', 'SPD'): {'panel1': 102.0, 'panel2': 149.5, 'panel3': np.nan},
              ('max', 'GRP'): {'panel1': 33, 'panel2': 45, 'panel3': 49},
              ('sum', 'SPD'): {'panel1': 102.0, 'panel2': 299.0, 'panel3': np.nan}}
    output = pd.DataFrame.from_dict(output)
    output.columns.names = ["_summary_type_", 'Variable']
    output.index.names = ['Panel']
    pd.testing.assert_frame_equal(raw_dp.long2wide(input_data, ["Panel"], "Metric", 'Metric_Value', {'mean': ['Grp', "Spend"], 'max': [
                                  'Grp'], 'sum': ["Spend"]}, variable_name='Variable', case_sensitive=False), output)

    # multiple summary type for given metric
    output = {('mean', 'GRP'): {'panel1': 33.0, 'panel2': 45.0, 'panel3': 36.5},
              ('mean', 'SPD'): {'panel1': 102.0, 'panel2': 149.5, 'panel3': np.nan},
              ('max', 'GRP'): {'panel1': 33, 'panel2': 45, 'panel3': 49}}
    output = pd.DataFrame.from_dict(output)
    output.columns.names = ["_summary_type_", 'Variable']
    output.index.names = ['Panel']
    pd.testing.assert_frame_equal(
        raw_dp.long2wide(
            input_data, ["Panel"], "Metric", 'Metric_Value', {
                'mean': [
                    'Grp', "Spend"], 'max': ['Grp']}, variable_name='Variable', case_sensitive=False), output)
