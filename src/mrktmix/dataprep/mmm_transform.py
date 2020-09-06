import numpy as np
import pandas as pd

from scipy.ndimage.interpolation import shift
from scipy.signal import lfilter


def apply_apl_(input_list, adstock, power, lag):
    """ Apply advertisement decay (carry over effect or decay effect), diminishing return and lag on array

    :param input_list: list of float with marketing or any other activities like spend
    :type input_list: list
    :param adstock: Adstock, carry over effect or decay effect on activity
    :type adstock: float
    :param power: Diminishing return or power transformation on activity
    :type power: float
    :param lag: Lag transformation on activity
    :type lag: float
    :return: list after applying adstock, power and lag transformation
    :rtype: list
    """

    return(shift(np.nan_to_num(np.power(lfilter([1], [1, -float(adstock)], input_list, axis=0), power)), lag, cval=0, order=1))


def create_base_(variable, date_input, freq, increasing=False, negative=False, periods=1, panel=None):
    """ Create dummy/base variable for modeling

    :param variable: Name of variable
    :type variable: str
    :param date_input: Dates used in base variable. If input type is str, periods and freq will be used to determine length of date.If
     input type is list, these dates will be used for base. If input type is tuple, the dates in the tuple will behave like range for date
    :type date_input: Union[str, list, tuple]
    :param freq: Frequency strings can have multiples, e.g. ‘5H’. See here for a list of pandas frequency aliases
    :type freq: str or DateOffset, default ‘D’
    :param increasing: Base values will in increaseing order if True. If false, it will be constant, defaults to False
    :type increasing: bool, optional
    :param negative: Controls the sign for base value, defaults to False
    :type negative: bool, optional
    :param periods: Controls number of period dates will be genrated. Will only be applicable when date_input is str, defaults to 1
    :type periods: int, optional
    :param panel: Create multiindex with panel on level 0, defaults to None
    :type panel: str, optional
    :return: pandas Series with date and panel index
    :rtype: pandas.Series
    """

    if isinstance(date_input, tuple):
        base_df_index = pd.date_range(start=date_input[0],
                                      end=date_input[1],
                                      freq=freq)
    elif isinstance(date_input, str):
        base_df_index = pd.date_range(start=date_input,
                                      periods=periods,
                                      freq=freq)
    else:
        base_df_index = pd.to_datetime(date_input)
    if panel is not None:
        base_df_index = pd.MultiIndex.from_tuples(list(zip([panel] * len(base_df_index), base_df_index)))
    base_df_index.freq = None
    base_df = pd.Series(1, index=base_df_index, name=variable)
    if increasing:
        base_df = base_df.cumsum()
    if negative:
        base_df = base_df * -1
    return(base_df)
