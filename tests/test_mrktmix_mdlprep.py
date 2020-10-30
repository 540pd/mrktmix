import numpy as np
import pandas as pd
import random

from mrktmix.dataprep import mdl_data as raw_dp

def test_generate_code():
    # test default value
    random.seed(999)
    assert raw_dp.generate_code()=='F85'
    # test with given code length 
    random.seed(999)
    assert raw_dp.generate_code(code_length=5)=='F854I'
    # test with given chars
    random.seed(999)
    assert raw_dp.generate_code(code_length=5,chars="ABC")=='CACCC'

def test_apply_mapping():
    # Case sensitive = True feature
    long_format_data={'Panel': {0: 'panel1', 1: 'panel1', 2: 'panel1', 3: 'panel1'},
     'Channel': {0: 'TV_Free', 1: 'TV_free', 2: 'TV', 3: 'TV'},
     'Metric': {0: 'GRp', 1: 'Spend', 2: 'GRP', 3: 'GRP'}}
    long_format_data=pd.DataFrame.from_dict(long_format_data)
    output_data={'Panel': {0: 'panel1', 1: 'panel1', 2: 'panel1', 3: 'panel1'},
     'Channel': {0: 'TV_Free', 1: 'TV_free', 2: 'TV', 3: 'TV'},
     'Metric': {0: 'GRp', 1: 'Spend', 2: 'GRP', 3: 'GRP'},
     '_Variable_': {0: 'ZOZ_4IU', 1: 'ERB_GMJ', 2: 'QDQ_F85', 3: 'QDQ_F85'}}
    output_data=pd.DataFrame.from_dict(output_data)
    output_dict={'GRP': 'F85', 'GRp': '4IU', 'Spend': 'GMJ', 'TV': 'QDQ', 'TV_Free': 'ZOZ', 'TV_free': 'ERB'}
    random.seed(999) 
    mapped_data,description_mapping = raw_dp.apply_mapping(long_format_data, ['Channel','Metric'], delimeter="_", description2code={}, case_sensitive=True)
    pd.testing.assert_frame_equal(mapped_data,output_data)
    assert output_dict==description_mapping
    
    # Case sensitive = False feature
    long_format_data={'Panel': {0: 'panel1', 1: 'panel1', 2: 'panel1', 3: 'panel1'},
     'Channel': {0: 'TV_Free', 1: 'TV_free', 2: 'TV', 3: 'TV'},
     'Metric': {0: 'GRp', 1: 'Spend', 2: 'GRP', 3: 'GRP'}}
    long_format_data=pd.DataFrame.from_dict(long_format_data)
    output_data={'Panel': {0: 'panel1', 1: 'panel1', 2: 'panel1', 3: 'panel1'},
     'Channel': {0: 'TV_Free', 1: 'TV_free', 2: 'TV', 3: 'TV'},
     'Metric': {0: 'GRp', 1: 'Spend', 2: 'GRP', 3: 'GRP'},
     '_Variable_': {0: 'ZOZ_F85', 1: 'ZOZ_GMJ', 2: 'QDQ_F85', 3: 'QDQ_F85'}}
    output_data=pd.DataFrame.from_dict(output_data)
    output_dict={'GRP': 'F85', 'GRp': 'F85', 'Spend': 'GMJ', 'TV': 'QDQ', 'TV_Free': 'ZOZ', 'TV_free': 'ZOZ'}
    random.seed(999) 
    mapped_data,description_mapping = raw_dp.apply_mapping(long_format_data, ['Channel','Metric'], delimeter="_", description2code={}, case_sensitive=False)
    pd.testing.assert_frame_equal(mapped_data,output_data)
    assert output_dict==description_mapping
    
    # Test different delimeter
    long_format_data={'Panel': {0: 'panel1', 1: 'panel1', 2: 'panel1', 3: 'panel1'},
     'Channel': {0: 'TV_Free', 1: 'TV_free', 2: 'TV', 3: 'TV'},
     'Metric': {0: 'GRp', 1: 'Spend', 2: 'GRP', 3: 'GRP'}}
    long_format_data=pd.DataFrame.from_dict(long_format_data)
    output_data={'Panel': {0: 'panel1', 1: 'panel1', 2: 'panel1', 3: 'panel1'},
     'Channel': {0: 'TV_Free', 1: 'TV_free', 2: 'TV', 3: 'TV'},
     'Metric': {0: 'GRp', 1: 'Spend', 2: 'GRP', 3: 'GRP'},
     '_Variable_': {0: 'ZOZ.F85', 1: 'ZOZ.GMJ', 2: 'QDQ.F85', 3: 'QDQ.F85'}}
    output_data=pd.DataFrame.from_dict(output_data)
    output_dict={'GRP': 'F85', 'GRp': 'F85', 'Spend': 'GMJ', 'TV': 'QDQ', 'TV_Free': 'ZOZ', 'TV_free': 'ZOZ'}
    random.seed(999) 
    mapped_data, description_mapping = raw_dp.apply_mapping(long_format_data, ['Channel','Metric'], delimeter=".", description2code={}, case_sensitive=False)
    pd.testing.assert_frame_equal(mapped_data,output_data)
    assert output_dict==description_mapping    
    
    # Test predefined mapping feature when case sensitive is true
    long_format_data={'Panel': {0: 'panel1', 1: 'panel1', 2: 'panel1', 3: 'panel1'},
     'Channel': {0: 'TV_Free', 1: 'TV_free', 2: 'TV', 3: 'TV'},
     'Metric': {0: 'GRp', 1: 'Spend', 2: 'GRP', 3: 'GRP'}}
    long_format_data=pd.DataFrame.from_dict(long_format_data)
    output_data={'Panel': {0: 'panel1', 1: 'panel1', 2: 'panel1', 3: 'panel1'},
     'Channel': {0: 'TV_Free', 1: 'TV_free', 2: 'TV', 3: 'TV'},
     'Metric': {0: 'GRp', 1: 'Spend', 2: 'GRP', 3: 'GRP'},
     '_Variable_': {0: 'GMJ.F85', 1: 'QDQ.4IU', 2: 'tv.TV_GRP', 3: 'tv.TV_GRP'}}
    output_data=pd.DataFrame.from_dict(output_data)
    output_dict={'GRP': 'TV_GRP',
     'TV': 'tv',
     'GRp': 'F85',
     'Spend': '4IU',
     'TV_Free': 'GMJ',
     'TV_free': 'QDQ'}
    random.seed(999) 
    mapped_data,description_mapping = raw_dp.apply_mapping(long_format_data, ['Channel','Metric'], delimeter=".", description2code={'GRP':'TV_GRP',"TV":"tv"}, case_sensitive=True)
    pd.testing.assert_frame_equal(mapped_data,output_data)
    assert output_dict==description_mapping
    
    # Test predefined mapping feature when case sensitive is false
    long_format_data={'Panel': {0: 'panel1', 1: 'panel1', 2: 'panel1', 3: 'panel1'},
     'Channel': {0: 'TV_Free', 1: 'TV_free', 2: 'TV', 3: 'TV'},
     'Metric': {0: 'GRp', 1: 'Spend', 2: 'GRP', 3: 'GRP'}}
    long_format_data=pd.DataFrame.from_dict(long_format_data)
    output_data={'Panel': {0: 'panel1', 1: 'panel1', 2: 'panel1', 3: 'panel1'},
     'Channel': {0: 'TV_Free', 1: 'TV_free', 2: 'TV', 3: 'TV'},
     'Metric': {0: 'GRp', 1: 'Spend', 2: 'GRP', 3: 'GRP'},
     '_Variable_': {0: '4IU.TV_GRP', 1: '4IU.F85', 2: 'tv.TV_GRP', 3: 'tv.TV_GRP'}}
    output_data=pd.DataFrame.from_dict(output_data)
    output_dict={'GRP': 'TV_GRP',
     'GRp': 'TV_GRP',
     'TV': 'tv',
     'Spend': 'F85',
     'TV_Free': '4IU',
     'TV_free': '4IU'}
    random.seed(999) 
    mapped_data,description_mapping = raw_dp.apply_mapping(long_format_data, ['Channel','Metric'], delimeter=".", description2code={'GRP':'TV_GRP',"TV":"tv","TV":"tv"}, case_sensitive=False)
    pd.testing.assert_frame_equal(mapped_data,output_data)
    assert output_dict==description_mapping

def test_long2wide():
    #input data
    input_data={'Panel': {0: 'panel1',
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
    input_data=pd.DataFrame.from_dict(input_data)

    # Test when summary type is case sensitive
    output={('mean', 'GRP'): {'panel1': 33.0, 'panel2': np.nan, 'panel3': np.nan},
     ('sum', 'GRP'): {'panel1': np.nan, 'panel2': 45.0, 'panel3': 73.0},
     ('sum', 'SPD'): {'panel1': 102.0, 'panel2': 299.0, 'panel3': np.nan}}
    output=pd.DataFrame.from_dict(output)
    output.columns.names=["_summary_type_",'Variable']
    output.index.names=['Panel']
    pd.testing.assert_frame_equal(raw_dp.long2wide(input_data, ["Panel"], "Metric", 'Metric_Value', {'mean':['GRp']}, variable_name='Variable', case_sensitive=True),output)
    
    # Test when summary type is not case sensitive
    output={('mean', 'GRP'): {'panel1': 33.0, 'panel2': 45.0, 'panel3': 36.5},
     ('sum', 'SPD'): {'panel1': 102.0, 'panel2': 299.0, 'panel3': np.nan}}
    output=pd.DataFrame.from_dict(output)
    output.columns.names=["_summary_type_",'Variable']
    output.index.names=['Panel']
    pd.testing.assert_frame_equal(raw_dp.long2wide(input_data, ["Panel"], "Metric", 'Metric_Value',{'mean':['GRp']}, variable_name='Variable', case_sensitive=False),output)

    # multiple summary type for given metric
    output={('mean', 'GRP'): {'panel1': 33.0, 'panel2': 45.0, 'panel3': 36.5},
     ('max', 'GRP'): {'panel1': 33, 'panel2': 45, 'panel3': 49},
     ('sum', 'SPD'): {'panel1': 102.0, 'panel2': 299.0, 'panel3': np.nan}}
    output=pd.DataFrame.from_dict(output)
    output.columns.names=["_summary_type_",'Variable']
    output.index.names=['Panel']
    pd.testing.assert_frame_equal(raw_dp.long2wide(input_data, ["Panel"], "Metric", 'Metric_Value', {'mean':['Grp'],'max':['Grp']}, variable_name='Variable', case_sensitive=False),output)
    
    # multiple summary type for given metric
    output={('mean', 'GRP'): {'panel1': 33.0, 'panel2': 45.0, 'panel3': 36.5},
     ('mean', 'SPD'): {'panel1': 102.0, 'panel2': 149.5, 'panel3': np.nan},
     ('max', 'GRP'): {'panel1': 33, 'panel2': 45, 'panel3': 49},
     ('sum', 'SPD'): {'panel1': 102.0, 'panel2': 299.0, 'panel3': np.nan}}
    output=pd.DataFrame.from_dict(output)
    output.columns.names=["_summary_type_",'Variable']
    output.index.names=['Panel']
    pd.testing.assert_frame_equal(raw_dp.long2wide(input_data, ["Panel"], "Metric", 'Metric_Value', {'mean':['Grp',"Spend"],'max':['Grp'],'sum':["Spend"]}, variable_name='Variable', case_sensitive=False),
        output)
    
    # multiple summary type for given metric
    output={('mean', 'GRP'): {'panel1': 33.0, 'panel2': 45.0, 'panel3': 36.5},
     ('mean', 'SPD'): {'panel1': 102.0, 'panel2': 149.5, 'panel3': np.nan},
     ('max', 'GRP'): {'panel1': 33, 'panel2': 45, 'panel3': 49}}
    output=pd.DataFrame.from_dict(output)
    output.columns.names=["_summary_type_",'Variable']
    output.index.names=['Panel']
    pd.testing.assert_frame_equal(raw_dp.long2wide(input_data, ["Panel"], "Metric", 'Metric_Value', {'mean':['Grp',"Spend"],'max':['Grp']}, variable_name='Variable', case_sensitive=False),output)
