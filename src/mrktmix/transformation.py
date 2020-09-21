import numpy as np
import pandas as pd

from mrktmix.dataprep import mmm_transform as dp


def create_base(variable, date_input, freq, increasing=False, negative=False, periods=1, panel=None):
    """ Create dummy/base variable for modeling

    :param variable: Name of variables
    :type variable: list
    :param date_input: List of string, list and tuble of Dates used in base variable. If input type is str, periods and freq will be used
     to determine length of date. If input type is list, these dates will be used for base. If input type is tuple, the dates in the tuple
     will behave like range for date
    :type date_input: list
    :param freq: list of frequency strings or frequency strings. It can have multiples, e.g. ‘5H’. See here for a list of pandas frequency
     aliases
    :type freq: Union[list, str]
    :param increasing: Base values will in increaseing order if True. If false, it will be constant, defaults to False
    :type increasing: Union[list, bool], optional
    :param negative: Controls the sign for base value, defaults to False
    :type negative: Union[list,bool], optional
    :param periods: Controls number of period dates will be genrated. Will only be applicable when date_input is str, defaults to 1
    :type periods: Union[list, int], optional
    :param panel: Create multiindex with panel on level 0, defaults to None
    :type panel: Union[list,str], optional
    :return: pandas DataFrame with date and panel index
    :rtype: pandas.DataFrame
    """
    if isinstance(variable, (np.ndarray, list)):
        df = pd.DataFrame({"variable": variable, "date_input": date_input, "freq": freq,
                           "increasing": increasing, "negative": negative, "periods": periods, "panel": panel})
        base_df = (df.apply(lambda row: dp.create_base_(row[0],
                                                        row[1],
                                                        row[2],
                                                        increasing=row[3],
                                                        negative=row[4],
                                                        periods=row[5],
                                                        panel=row[6]),
                            axis=1) .set_index(df.iloc[:,
                                                       0].values) .sum(level=0,
                                                                       min_count=1) .transpose())
        base_df.index.freq = None
    else:
        base_df = dp.create_base_(variable, date_input, freq, increasing=increasing, negative=negative, periods=periods, panel=panel)
    return(base_df)


def apply_apl(dframe, dict_apl):
    """ Apply advertisement decay (carry over effect or decay effect), diminishing return and lag on pandas.DataFrame

    :param dframe: DataFrame with marketing or any other activities like spend
    :type dframe: pandas.DataFrame
    :param dict_apl: Dictionary with list of Adstock, carry over effect or decay effect on activity. Key of the dictionary can be name of
        panel or Bool. If key is true, then transformation is applied at panel level. If key is false, then transformation is applied on
        entire dataframe. If trasformation needs to be applied at panel level (name of panel should at level -2 in multiindex row
    :type dict_apl: dictionary with list values. Keys can be string or bool
    :return: Returns input after applying adstock, power and lag transformation
    :rtype: pandas.DataFrame
    """
    if (len(dict_apl) == 1) and (False in dict_apl.keys()):
        all_vars = dict_apl[False]
        df_transformed = pd.concat([dp.apply_apl_series(dframe[var[0]], var[1], var[2], var[3]) for var in all_vars], axis=1)
    elif (len(dict_apl) == 1) and (True in dict_apl.keys()):
        all_vars = dict_apl[True]
        df_grouped = dframe.groupby(dframe.droplevel(-1).index)
        df_transformed = pd.concat([df_grouped[var[0]].transform(
            lambda x: dp.apply_apl_series(x, var[1], var[2], var[3])) for var in all_vars], axis=1)
        df_transformed.columns = pd.MultiIndex.from_tuples(all_vars)
    else:
        df_transformed = pd.DataFrame()
        for panel, all_vars in dict_apl.items():
            df_transformed = pd.concat([df_transformed, pd.concat([dp.apply_apl_series(
                dframe.loc[[panel], var[0]], var[1], var[2], var[3]) for var in all_vars], axis=1)])
    df_transformed.columns.names = ["Variable", "Adstock", "Power", "Lag"]
    return(df_transformed)


def apply_coef(raw_data, coef, dep_series):
    """ Apply coefficient and transformation on raw data

    :param raw_data: modeling dataframe with date index at level -1. If panel is present, it should be at level -2.
    :type raw_data: pandas.DataFrame
    :param coef: Coefficient and parameter to be applied on modeling dataframe. Coefficient should have tuple of variable, adstock,power
        and lag at index level -1. If panel is present in modeling data, then coefficient must have panel information in index at level -2.
    :type coef: pandas.Series
    :param dep_series: Dependent Series will be used to calculate residual. It must be at same level modeling dataframe
    :type dep_series: pandas.Series or None
    :return: Decomposition of dependent series
    :rtype: pandas.DataFrame
    """
    panel_var = (pd.Series(coef.index.get_level_values(-1))
                 .groupby(coef.droplevel(-1).index)
                 .apply(list)
                 .to_dict())
    after_apl = apply_apl(raw_data, panel_var)
    coef2frame = coef.unstack()
    coef2frame.columns = pd.MultiIndex.from_tuples(coef2frame.columns)
    dep_decomposition = after_apl[coef2frame.columns].mul(coef2frame.reindex(after_apl.index, level=0))
    if dep_series is not None:
        dep_decomposition[("Residual", 0, 1, 0)] = dep_series-dep_decomposition.sum(axis=1)
    return(dep_decomposition)


def collapse_date(dep_decompose, date_dict):
    """ Summarise data after collapsing date (index at level -1). Summarization is based on date dictionary given in input.

    :param dep_decompose: data to be summarised. date should be index with level -1.
    :type dep_decompose: pandas.DataFrame
    :param date_dict: dictionary with name and tuples of dates. Dates are inclusive.
    :type date_dict: dictionary
    :return: Summarise after collpasing date
    :rtype: pandas.DataFrame
    """
    all_decomp_smry = []
    for key, val in date_dict.items():
        selected_date = [upp_lim == low_lim for upp_lim, low_lim in zip(
            val[1] >= dep_decompose.index.get_level_values(-1), dep_decompose.index.get_level_values(-1) >= val[0])]
        if dep_decompose.index.nlevels-1:
            decomp_smry = dep_decompose.iloc[selected_date].sum(level=range(0, dep_decompose.index.nlevels-1))
        else:
            decomp_smry = dep_decompose.iloc[selected_date].sum()
        if isinstance(decomp_smry, pd.Series):
            decomp_smry = decomp_smry.rename(key)
        else:
            decomp_smry = decomp_smry.stack(list(range(0, dep_decompose.columns.nlevels))).rename(key)
        if len(all_decomp_smry):
            all_decomp_smry = pd.concat([all_decomp_smry, decomp_smry], axis=1)
        else:
            all_decomp_smry = decomp_smry.to_frame()
    return(all_decomp_smry)
