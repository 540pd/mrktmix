import numpy as np
import pandas as pd

from mrktmix.dataprep import mdl_data as dc

def create_mdldata(input_files, panel_variables, description_variables, metric_value, metric_index=-1, description2code={}, summary_type={}, delimeter="_", code_length=3, case_sensitive=False, iteratively=False):
    """
    Covert list of input files into modeling data.

    :param input_files: dictionary with filename and data 
    :type input_files: dictionary with filename as key and pandas.DataFrame as values
    :param panel_variables: name of panel variables in mapped data
    :type panel_variables: list of string
    :param description_variables: name of description variables in mapped data
    :type description_variables: list of string
    :param metric_value: name of metric value (summary variable) in mapped data
    :type metric_value: string
    :param metric_index: index representing metric variable in description variables
    :type metric_index: int
    :param description2code: mapping in dictionary format with description as key and code as value, defult is empty dictionary
    :type description2code: dictionary
    :param summary_type: summary function to be applied on metric variable. key of summary type is function and 
     value of summary type is list of metric values. default value is empty
    :type summary_type: dictionary
    :param delimeter: delimeter used inbetween description mapping code
    :type delimeter: string    
    :param code_length: length of code to be geneated, defualt value is 3
    :type code_length: int
    :param case_sensitive: whether mapping to code is case sensitive or not
    :type case_sensitive: bool
    :param iteratively: whether modeling data processing should be done filewise or not (append data and process it in single pass) 
    :type iteratively: bool    
    :return: modeling dataframe with panel variable as row index and variable_name as variable. Second item of tuple represents mapping from description to code. Third item of tuple represents mapping from file to variables
    :rtype: tuples of pandas.DataFrame, dictionary, dictionary
    """

    description2code_agg=description2code.copy()
    file_mapping=pd.DataFrame()
    if iteratively:
        mdl_data=pd.DataFrame()
        for name_df, input_df in input_files.items():
            mapped_data, des2code = dc.apply_mapping(input_df, description_variables,code_length=code_length, delimeter=delimeter, description2code=description2code_agg, case_sensitive=case_sensitive)
            description2code_agg.update(des2code)
            mapped_data["_File_"]=name_df
            file_mapping=file_mapping.append(mapped_data[["_File_", "_Variable_",*description_variables]].drop_duplicates().set_index(["_File_","_Variable_"]))
            mdl_df=dc.long2wide(mapped_data, panel_variables, description_variables[metric_index], metric_value, summary_type, variable_name='_Variable_', case_sensitive=case_sensitive)
            mdl_data=pd.concat([mdl_data, mdl_df], axis=1)
    else:
        input_df=pd.concat(input_files)
        input_df=input_df.reset_index(level=0).rename(columns={'level_0':'_File_'})
        mapped_data, des2code = dc.apply_mapping(input_df, description_variables, code_length=code_length, delimeter=delimeter, description2code=description2code_agg, case_sensitive=case_sensitive)
        description2code_agg.update(des2code)
        file_mapping=file_mapping.append(mapped_data[["_File_", "_Variable_",*description_variables]].drop_duplicates())
        file_mapping=file_mapping.set_index(["_File_","_Variable_"])
        mdl_data=dc.long2wide(mapped_data, panel_variables, description_variables[metric_index], metric_value, summary_type, variable_name='_Variable_', case_sensitive=case_sensitive)
    return(mdl_data, description2code_agg, file_mapping)

def spread_notna(input_df, prior=True):
    """
    Distributes non missing values in data frame to prior missing or post missing values equally

    :param input_df: input dataframe where adjustment needs to made 
    :type input_df: pandas.DataFrame
    :param prior: whether prior or post adjustment needs to made. If prior is Ture, adjustment of non missing is spread toward preceding missing values. If its false, then non missing is spread toward succeeding missing values. 
    :type prior: bool
    :return: Adjusted dataframe
    :rtype: pandas.DataFrame
    """

    if prior:
        shift_factor=1
    else:
        shift_factor=-1
    change_track=input_df.isnull()
    adj_factor=(change_track.ne(change_track.shift(shift_factor))
     .cumsum()
     .apply(lambda x: x.map(x.value_counts()))
     .where(change_track)
     .shift(shift_factor)
     .add(1)
     .fillna(1))
    if prior:
        return(input_df.div(adj_factor).bfill())
    else:
        return(input_df.div(adj_factor).ffill())