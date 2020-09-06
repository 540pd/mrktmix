import numpy as np
import pandas as pd

from mrktmix.dataprep import mmm_transform as dp


def apply_apl(df_series, adstock, power, lag):
    """ Apply advertisement decay (carry over effect or decay effect), diminishing return and lag on pandas.Series

    :param df_series: Series with marketing or any other activities like spend
    :type input_list: pandas.Series
    :param adstock: Adstock, carry over effect or decay effect on activity
    :type adstock: float
    :param power: Diminishing return or power transformation on activity
    :type power: float
    :param lag: Lag transformation on activity
    :type lag: float
    :return: Returns input after applying adstock, power and lag transformation
    :rtype: pandas.Series
    """
    if isinstance(df_series, pd.Series):
        return(pd.Series(dp.apply_apl_(df_series, adstock, power, lag), index=df_series.index, name=(df_series.name, adstock, power, lag)))


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
