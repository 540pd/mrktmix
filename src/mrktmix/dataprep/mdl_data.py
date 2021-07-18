import random

import numpy as np
import pandas as pd
from scipy.stats import pareto


def generate_code(code_length, chars='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', cum_weights=None):
    """
    Generate random code from ascii and digits based character length

    :param code_length: length of code to be generated.
    :type code_length: int
    :param chars: characters from which random literals will be selected
    :type chars: string
    :return: random slected literal of length given in code length from chars
    :rtype: string
    """
    return (''.join(random.choices(chars, cum_weights=cum_weights, k=code_length)))


def update_mapping(new_description, user_defined_mapping, case_sensitive=True):
    """
    Subset relevant mapping from user defined mapping based on new description. Also handles if the mapping are case sensitive
    or not

    :param new_description: array to generate mapping
    :type new_description: numpy.darray
    :param user_defined_mapping: pre-defined mapping
    :type user_defined_mapping: dictionary
    :param case_sensitive: whether the relevant mapping on case or not
    :type case_sensitive: bool
    :return: relevant dictionary based on new description
    :rtype: dictionary
    """

    des2code = {}
    if not case_sensitive:
        description2code_title = {k.title(): v for k, v in user_defined_mapping.items()}
        [des2code.update({(i, description2code_title[i.title()])}) for i in new_description if i.title() in list(description2code_title)]
    else:
        des2code.update({k: user_defined_mapping[k] for k in new_description[np.isin(new_description, list(user_defined_mapping.keys()))]})
    return(des2code)


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

    df4mdl = long_format_data.copy()
    # trim and convert description columns to string
    df4mdl[description_variables] = df4mdl[description_variables].astype(str).apply(lambda x: x.str.strip())
    # Description present in supplied mapping dictionary
    new_description = np.unique(df4mdl[description_variables].values)
    # subset relevant mapping
    des2code = {}
    # correct any inconsistency in given description. same description cann't have two or more codes
    if len(description2code):
        des2code.update(update_mapping(new_description, description2code, case_sensitive=case_sensitive))

    # Generate new mapping if description is not present in supplied mapping dictionary
    new_description = new_description[~np.isin(new_description, list(des2code.keys()))]
    if len(new_description):
        des2code_new = {}
        avoid_infinite_loop = 1
        # generate code
        if not case_sensitive:
            series4code_original = pd.Series(new_description, index=new_description).replace(r'\W|_', '', regex=True).str.upper()
            duplicate_description = series4code_original.index.str.upper().duplicated()
            series4code = series4code_original[~duplicate_description]
        else:
            series4code = pd.Series(new_description, index=new_description).replace(r'\W|_', '', regex=True).str.upper()
        # distribution of literal- is used to generate code
        dist_fixed = pareto.cdf(range(1, series4code.str.len().max() + 2), 1)
        while True:
            if avoid_infinite_loop == 1:
                des2code_generate = (series4code * ((code_length / series4code.str.len()).apply(np.ceil)
                                                    ).astype(int)).str[:code_length]
            elif avoid_infinite_loop < 5:
                des2code_generate = series4code.apply(lambda x: generate_code(code_length, x, dist_fixed[1:len(x) + 1]))
            elif avoid_infinite_loop < 8:
                des2code_generate = series4code.apply(lambda x: generate_code(code_length, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            else:
                des2code_generate = series4code.apply(lambda x: generate_code(code_length, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
            # check if code is duplicate or exists in original description to code mapping
            des_code_dup = des2code_generate.isin(
                des2code.values()) | des2code_generate.duplicated(
                keep=False) | des2code_generate.isin(
                des2code_new.values())
            # Finalize mapping if duplicate is not present
            des2code_new.update(des2code_generate[~des_code_dup].to_dict())
            # break loop if no new description is present
            if not(des_code_dup.sum()):
                break
            # update new description if duplicate codes are present
            avoid_infinite_loop = avoid_infinite_loop + 1
            # update series to generate code
            series4code = series4code[des_code_dup.values]
        if not case_sensitive:
            des2code_new.update(
                update_mapping(
                    series4code_original[duplicate_description].index,
                    des2code_new,
                    case_sensitive=case_sensitive))
        des2code.update(des2code_new)

    # Create Variable in raw file
    df4mdl["_Variable_"] = (df4mdl[description_variables]
                            .replace(des2code)
                            .apply(lambda x: delimeter.join(x), axis=1))
    return(df4mdl, des2code)


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

    df4mdl = mapped_data.copy()
    if not case_sensitive:
        # convert all metric to title case
        df4mdl[metric_var] = df4mdl[metric_var].str.title()
        # convert all values of dictionary to title case
        summary_type = {k: [i.title() for i in v] for k, v in summary_type.items()}
    # modeling data
    mdl_data = pd.DataFrame()
    # modeling data where summary type is present
    for k, v in summary_type.items():
        mdl_df = pd.pivot_table(df4mdl[df4mdl[metric_var].isin(v)].assign(_summary_type_=k),
                                values=value_variable,
                                index=panel_variables,
                                columns=["_summary_type_", variable_name],
                                aggfunc=k)
        mdl_data = pd.concat([mdl_data, mdl_df], axis=1)
    # modeling data where summary type is not present
    mdl_df = pd.pivot_table(df4mdl[~df4mdl[metric_var].isin([j for i in summary_type.values() for j in i])].assign(_summary_type_='sum'),
                            values=value_variable,
                            index=panel_variables,
                            columns=["_summary_type_", variable_name],
                            aggfunc='sum')
    mdl_data = pd.concat([mdl_data, mdl_df], axis=1)
    mdl_data.index.names = mdl_df.index.names
    return(mdl_data)
