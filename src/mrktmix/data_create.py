from collections import defaultdict

import numpy as np
import pandas as pd

from mrktmix.dataprep import mdl_data as dc


def create_mdldata(
        input_files,
        panel_variables,
        description_variables,
        metric_value,
        metric_index=-1,
        description2code={},
        summary_type={},
        delimeter="_",
        code_length=3,
        case_sensitive=False,
        iteratively=False,
        verbose=False):
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
    :return: modeling dataframe with panel variable as row index and variable_name as variable. Second item of tuple represents
    mapping from description to code. Third item of tuple represents mapping file (description of variable)
    :rtype: tuples of pandas.DataFrame, dictionary, pandas.DataFrame
    """

    description2code_agg = description2code.copy()
    file_mapping = pd.DataFrame()
    if iteratively:
        mdl_data = pd.DataFrame()
        for name_df, input_df in input_files.items():
            if verbose:
                print(name_df)
            mapped_data, des2code = dc.apply_mapping(
                input_df,
                description_variables,
                code_length=code_length,
                delimeter=delimeter, description2code=description2code_agg,
                case_sensitive=case_sensitive)
            description2code_agg.update(des2code)
            mapped_data["_File_"] = name_df
            file_mapping = file_mapping.append((
                mapped_data[["_File_", "_Variable_", *description_variables]]
                .drop_duplicates()
                .set_index(["_File_", "_Variable_"])))
            mdl_df = dc.long2wide(
                mapped_data,
                panel_variables,
                description_variables[metric_index],
                metric_value,
                summary_type,
                variable_name='_Variable_',
                case_sensitive=case_sensitive)
            mdl_data = pd.concat([mdl_data, mdl_df], axis=1)
    else:
        input_df = pd.concat(input_files)
        input_df = input_df.reset_index(level=0).rename(columns={'level_0': '_File_'})
        mapped_data, des2code = dc.apply_mapping(input_df, description_variables, code_length=code_length,
                                                 delimeter=delimeter, description2code=description2code_agg, case_sensitive=case_sensitive)
        description2code_agg.update(des2code)
        file_mapping = file_mapping.append(mapped_data[["_File_", "_Variable_", *description_variables]].drop_duplicates())
        file_mapping = file_mapping.set_index(["_File_", "_Variable_"])
        mdl_data = dc.long2wide(
            mapped_data,
            panel_variables,
            description_variables[metric_index],
            metric_value,
            summary_type,
            variable_name='_Variable_',
            case_sensitive=case_sensitive)
    return(mdl_data, description2code_agg, file_mapping)


def spread_notna(input_df, prior=True):
    """
    Distributes non missing values in data frame to prior missing or post missing values equally

    :param input_df: input dataframe where adjustment needs to made
    :type input_df: pandas.DataFrame
    :param prior: whether prior or post adjustment needs to made. If prior is Ture, adjustment of non missing is spread toward
    preceding missing values. If its false, then non missing is spread toward succeeding missing values.
    :type prior: bool
    :return: Adjusted dataframe
    :rtype: pandas.DataFrame
    """

    if prior:
        shift_factor = 1
    else:
        shift_factor = -1
    change_track = input_df.isnull()
    adj_factor = (change_track.ne(change_track.shift(shift_factor))
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


def parse_variable(Name_series, ids_index, delimiter="_", anti=False):
    """
    Get new id/code based on id_index after splitting on given delimeter.

    :param Name_series: pandas series values will be splitted based on delimiter to find relevent ids index
    :type Name_series: pandas.Series
    :param ids_index: index in Name_series to be extracted after splitting by delimiter
    :type ids_index: list of integer
    :param delimiter: delimiter used in Name_series
    :type delimiter: string
    :param anti: index representing metric variable in description variables
    :type anti: bool
    :return: return matching ids after spliting by delimiter on Name_series
    :rtype: pandas.Series
    """

    if max(ids_index) < 0:
        var_split = Name_series.str.rsplit(pat=delimiter, n=max(map(abs, ids_index)), expand=True)
    else:
        var_split = Name_series.str.split(pat=delimiter, expand=True)
    var_split[var_split.isna()] = ""
    if anti:
        var_split['Variable'] = var_split.drop(var_split.columns[ids_index], axis=1).apply(lambda x: delimiter.join(x), axis=1)
    else:
        var_split['Variable'] = var_split.filter(var_split.columns[ids_index], axis=1).apply(lambda x: delimiter.join(x), axis=1)
    var_split['Variable'] = var_split['Variable'].str.rstrip(delimiter)
    return (var_split['Variable'])


def aggregate_rows(Dataset, metric2smry_func, metric_column, value_column, case_senstive=False, print_suffix="\t"):
    """
    Returns the data frame file after removing all the duplicates & grouping it according to metric summary dictionary

    :param Dataset: input dataset to be aggregated
    :type Dataset: pandas.DataFrame
    :param metric2smry_func: dictionary with keys as metric columns and summary function as values
    :type metric2smry_func: Dictionary
    :param metric_column: Name of Metric column on which metric2smry_func will match keys for aggregation
    :type metric_column: String
    :param value_column: Numeric variable in Dataset on which summary will be applied
    :type value_column: String
    :param case_senstive: whether keys of metric2smry will case senstive match or not
    :type case_senstive: bool
    :param print_suffix: space between first column and message printed
    :type print_suffix: string
    :return: aggregated data after applying metric2smry_func on metric_column and summarise value_column
    :rtype: pandas.DataFrame
    """

    # Find if aggregation is required in data
    duplicate_ids = Dataset.duplicated(subset=[*Dataset.columns.difference([value_column])], keep=False)
    # Aggregation of Data with duplicate ids
    df_duplicates = Dataset[duplicate_ids].copy()
    if df_duplicates.shape[0] != 0:
        if not case_senstive:
            metric2smry_func.update({k.title(): v for k, v in metric2smry_func.items()})
            [metric2smry_func.update({(i, metric2smry_func[i.title()])}) for i in df_duplicates[metric_column].unique()
             if i.title() in list(metric2smry_func)]  # same code without considering case
        print(print_suffix + "Shape of duplicate record is", df_duplicates.shape)
        smry_fun_navl = ~np.isin(df_duplicates[metric_column].unique(), list(metric2smry_func.keys()))
        if sum(smry_fun_navl):
            print("List of metric summary function not available in input dictionary. Please update it.",
                  "\n\t", df_duplicates[metric_column].unique()[smry_fun_navl],)
            raise Exception('Please update mapping dictionary.')
        else:
            df_duplicates["smry_func"] = df_duplicates[metric_column].replace(metric2smry_func)
            df_duplicates = (pd.concat([df.groupby([*Dataset.columns.difference([value_column]),
                                                    "smry_func"]).agg(smry_f) for smry_f,
                                        df in df_duplicates[[*Dataset.columns.difference([value_column]),
                                                             value_column,
                                                             "smry_func"]].groupby("smry_func")],
                                       sort=False) .reset_index() .drop(["smry_func"],
                                                                        axis=1))
    return pd.concat([Dataset[~duplicate_ids], df_duplicates], sort=False)


def reserve_dict(inputdict, concatenate="|"):
    '''
    Inserve dictionary. If there are duplicate values then it concatenate string to combine values

    :param inputdict: dictionary to be reversed
    :type inputdict: dictionary
    :param concatenate: string to be used if there are duplicate values in dictionary to concatenate
    values of reversed dictionary. Default value is '|'. If concatenate is None, values will be in form of list
    :type concatenate: string
    :return: aggregated data after applying metric2smry_func on metric_column and summarise value_column
    :rtype: dictionary
    '''
    reversed_dict = defaultdict(list)
    for key, value in inputdict.items():
        reversed_dict[value].append(key)
    if concatenate is None:
        return(dict(reversed_dict))
    else:
        return({key: concatenate.join(value) for key, value in reversed_dict.items()})


def update_description(description_input, new_code, new_description, delimeter="_", index=[], old_code=[]):
    '''
    Update description on index by replacing old code by new code and new description.

    :param description_df: description file with variable name in index and description on columns
    :type description_df: pd.DataFrame
    :param new_code: new code will be used inplace of old code in variable name
    :type new_code: string
    :param new_description: new description will be used inplace of corresponding column in description_df
    :type new_description: string
    :param delimeter: delimeters used in between code in variable name
    :type delimeter: string
    :param index: index of variable where update will be applied after spliting on delimeter. Default value is empty list. When
    empty, update will apply on all index
    :type index: list of integers
    :param old_code: list of old codes present in variable which needs to be replaced
    :type old_code: list of string
    :return: tuple of dictionary representing old variable to new variable name and updated description data
    :rtype: tuple of dictionary and pd.DataFrame
    '''
    mapping = description_input.index.to_series().str.split("_", expand=True)
    col_rename = dict(zip(description_input.columns[0:mapping.shape[1]], range(0, mapping.shape[1])))
    description_df = description_input.rename(columns=col_rename)
    if len(old_code) and len(index):
        description_df[mapping.iloc[:, index].isin(old_code)] = new_description
        mapping.iloc[:, index] = mapping.iloc[:, index].replace(to_replace=old_code, value=new_code)
    elif not len(old_code):
        description_df.iloc[:, index] = new_description
        mapping.iloc[:, index] = new_code
    elif not len(index):
        description_df[mapping.isin(old_code)] = new_description
        mapping = mapping.replace(to_replace=old_code, value=new_code)
    mapping = mapping.apply(lambda x: "_".join(x), axis=1).to_dict()
    description_df = description_df.rename(columns=reserve_dict(col_rename), index=mapping)
    description_df = description_df[~description_df.index.duplicated()]
    return (mapping, description_df)
