import numpy as np
import pandas as pd
import random

def generate_code(code_length=3, chars='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'): 
    """
    Generate random code from ascii and digits based character length
    
    :param code_length: length of code to be generated. default value is 3
    :type code_length: int
    :param chars: characters from which random literals will be selected
    :type chars: string
    :return: random slected literal of length given in code length from chars
    :rtype: string
    """
    return ''.join(random.choice(chars) for x in range(code_length))

def apply_mapping(long_format_data, description_variables, code_length=3, delimeter="_", description2code={}, case_sensitive=False):
    """
    Create/apply mapping on description variable on data to create variable name for modeling data

    :param long_format_data: dataframe to be used to create modeling data 
    :type long_format_data: pandas.DataFrame
    :param description_variables: columns in dataframe for which mapping codes needs to generated
    :type description_variables: list of string
    :param code_length: length of code to be geneated, defualt value is 3
    :type code_length: int
    :param delimeter: delimeter used inbetween description mapping code
    :type delimeter: string
    :param description2code: mapping in dictionary format with description as key and code as value, defult is empty dictionary
    :type description2code: dictionary
    :param case_sensitive: whether mapping to code is case sensitive or not
    :type case_sensitive: bool
    :return: dataframe after applying mapping code with column name "_Variable_". Also return mapping of description and code
    :rtype: tuple of pandas.DataFrame and dictionary
    """

    df4mdl=long_format_data.copy()
    # Description present in supplied mapping dictionary
    new_description=np.unique(df4mdl[description_variables].values)
    #subset relevant mapping
    des2code={}
    #same code without considering case
    if not case_sensitive:
        description2code_title={k.title(): v for k, v in description2code.items()}
        [des2code.update({(i,description2code_title[i.title()])}) for i in new_description if i.title() in list(description2code_title)] 
    # update dictionary if case matches
    des2code.update({k: description2code[k] for k in new_description[np.isin(new_description,list(description2code.keys()))]})

    # Generate new mapping if description is not present in supplied mapping dictionary
    new_description=new_description[~np.isin(new_description,list(des2code.keys()))] 
    des2code_keep={}
    if len(new_description):
        des2code_new={}
        while True:
            # generate code
            des2code_generate=pd.Series(np.nan, index=new_description).apply(lambda x: generate_code(code_length=code_length))
            # check if code is duplicate or exists in original description to code mapping
            des_code_dup = des2code_generate.isin(des2code.values()) | des2code_generate.duplicated(keep=False)
            # Finalize mapping if duplicate is not present
            des2code_new.update(des2code_generate[~des_code_dup].to_dict())
            # update new description if duplicate codes are present
            new_description =des_code_dup[des_code_dup].index
            # break loop if no new description is present
            if not(des_code_dup.sum()):
                break
        if not case_sensitive:
            # Code should be common if description is matches after ignoring case
            des2code_new=dict([(x,v) if x.lower() not in des2code_keep and not des2code_keep.update({x.lower():v}) else (x,des2code_keep[x.lower()]) for x, v in des2code_new.items()])
        des2code.update(des2code_new)
        
    # Create Variable in raw file
    df4mdl["_Variable_"]=(df4mdl[description_variables]
     .replace(des2code)
     .apply(lambda x: delimeter.join(x), axis=1))
    return(df4mdl,des2code)

def long2wide(mapped_data, panel_variables, metric_var, value_variable, summary_type={}, variable_name='_Variable_', case_sensitive=False):
    """
    Covert long format data into modeling data.

    :param mapped_data: dataframe to be converted to modeling data 
    :type mapped_data: pandas.DataFrame
    :param panel_variables: name of panel variables in mapped data
    :type panel_variables: list of string
    :param metric_var: name of metric variables in mapped data
    :type metric_var: string
    :param metric_value: name of metric value (summary variable) in mapped data
    :type metric_value: string
    :param summary_type: summary function to be applied on metric variable. key of summary type is function and 
     value of summary type is list of metric values. default value is empty
    :type summary_type: dictionary
    :param variable_name: variable name in mapped data to be mapped to modeling data
    :type variable_name: string
    :param case_sensitive: whether mapping to code is case sensitive or not
    :type case_sensitive: bool
    :return: modeling dataframe with panel variable as row index and variable_name as variable
    :rtype: pandas.DataFrame
    """

    df4mdl=mapped_data.copy()
    if not case_sensitive:
        # convert all metric to title case
        df4mdl[metric_var]=df4mdl[metric_var].str.title()
        # convert all values of dictionary to title case
        summary_type={k: [i.title() for i in v] for k, v in summary_type.items()}
    # modeling data
    mdl_data=pd.DataFrame()
    # modeling data where summary type is present
    for k, v in summary_type.items():
        mdl_df=pd.pivot_table(df4mdl[df4mdl[metric_var].isin(v)].assign(_summary_type_= k),
                       values=value_variable,
                       index=panel_variables,
                       columns=["_summary_type_",variable_name],
                       aggfunc=k)
        mdl_data=pd.concat([mdl_data,mdl_df], axis=1)
    # modeling data where summary type is not present
    mdl_df=pd.pivot_table(df4mdl[~df4mdl[metric_var].isin([j for i in summary_type.values() for j in i])].assign(_summary_type_= 'sum'),
                   values=value_variable,
                   index=panel_variables,
                   columns=["_summary_type_",variable_name],
                   aggfunc='sum')
    mdl_data=pd.concat([mdl_data,mdl_df], axis=1)
    mdl_data.index.names=mdl_df.index.names
    return(mdl_data)

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
            mapped_data, des2code = apply_mapping(input_df, description_variables,code_length=code_length, delimeter=delimeter, description2code=description2code_agg, case_sensitive=case_sensitive)
            description2code_agg.update(des2code)
            mapped_data["_File_"]=name_df
            file_mapping=file_mapping.append(mapped_data[["_File_", "_Variable_",*description_variables]].drop_duplicates().set_index(["_File_","_Variable_"]))
            mdl_df=long2wide(mapped_data, panel_variables, description_variables[metric_index], metric_value, summary_type, variable_name='_Variable_', case_sensitive=case_sensitive)
            mdl_data=pd.concat([mdl_data, mdl_df], axis=1)
    else:
        input_df=pd.concat(input_files)
        input_df=input_df.reset_index(level=0).rename(columns={'level_0':'_File_'})
        mapped_data, des2code = apply_mapping(input_df, description_variables, code_length=code_length, delimeter=delimeter, description2code=description2code_agg, case_sensitive=case_sensitive)
        description2code_agg.update(des2code)
        file_mapping=file_mapping.append(mapped_data[["_File_", "_Variable_",*description_variables]].drop_duplicates())
        file_mapping=file_mapping.set_index(["_File_","_Variable_"])
        mdl_data=long2wide(mapped_data, panel_variables, description_variables[metric_index], metric_value, summary_type, variable_name='_Variable_', case_sensitive=case_sensitive)
    return(mdl_data, description2code_agg, file_mapping)