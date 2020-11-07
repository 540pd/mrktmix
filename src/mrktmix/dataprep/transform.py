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


def apply_apl_series(df_series, adstock, power, lag):
    """ Apply advertisement decay (carry over effect or decay effect), diminishing return and lag on pandas.Series

    :param df_series: Series with marketing or any other activities like spend
    :type df_series: pandas.Series
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
        return(pd.Series(apply_apl_(df_series, adstock, power, lag), index=df_series.index, name=(df_series.name, adstock, power, lag)))


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

def segregate_variable(aggregated_data, segregated_data, match_sum=True):
    """
    Segregate aggregate variable in aggregated data based on proportion in segregated data. It assumes row index in aggregated and segregated data
    is same
    
    :param aggregated_data: Aggregated series with aggregate variable
    :type aggregated_data: pandas.Series
    :param segregated_data: Segregated data is used to compute proportion for segregation of aggregated data 
    :type segregated_data: pandas.DataFrame
    :param match_sum: When proportion calculated in segregated data is 0 and corresponding value is present in aggregated data,
    then sum of aggregated data won't match segregated output data. If match_sum is True, values in segregated output data is 
    adjusted to match its sum with sum of aggregated data 
    :type match_sum: bool
    :return: segregated data at variable level
    :rtype: pandas.DataFrame  
    """

    prop_data=segregated_data.div(segregated_data.sum(axis=1),axis=0)
    segregate_df=prop_data.mul(aggregated_data,axis=0)
    if match_sum:
        adj_total=(aggregated_data-segregate_df.sum(axis=1)).sum()
        segregate_df_final=(segregate_df+segregate_df/(segregate_df.sum().sum())*adj_total)
    else:
        segregate_df_final=segregate_df
    return(segregate_df_final)

def segregate_panel(aggregated_data, segregated_data, match_sum=True):
    """
    Segregate aggregate data at panel level based on proportion in segregated data. It assumes row index in aggregated and segregated data
    is same
    
    :param aggregated_data: Aggregated data at panel level
    :type aggregated_data: pandas.DataFrane
    :param segregated_data: Segregated data is used to compute proportion for segregation of aggregated data 
    :type segregated_data: pandas.DataFrame
    :param match_sum: When proportion calculated in segregated data is 0 and corresponding value is present in aggregated data,
    then sum of aggregated data won't match segregated output data. If match_sum is True, values in segregated output data is 
    adjusted to match its sum with sum of aggregated data 
    :type match_sum: bool
    :return: segregated data at panel level
    :rtype: pandas.DataFrame  
    """
    
    aggregate_df = segregated_data.sum(level = -1)
    prop_data=segregated_data.divide(aggregate_df.reindex(segregated_data.index, level= -1),axis=0)    
    segregate_df=aggregated_data.reindex(prop_data.index, level=-1).mul(prop_data)
    if match_sum:
        adj_series=(aggregated_data.sum()-segregate_df.sum())
        segregate_df_final=(segregate_df/segregate_df.sum()).mul(adj_series).add(segregate_df)
    else:
        segregate_df_final=segregate_df
    return(segregate_df_final)

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

    if max(ids_index)<0:
        var_split=Name_series.str.rsplit(pat=delimiter,n=max(map(abs, ids_index)),expand=True)
    else:
        var_split=Name_series.str.split(pat=delimiter,expand=True)
    var_split[var_split.isna()]=""
    if anti:
        var_split['Variable'] = var_split.drop(var_split.columns[ids_index], axis = 1).apply(lambda x: delimiter.join(x), axis=1)
    else:
        var_split['Variable'] = var_split.filter(var_split.columns[ids_index], axis = 1).apply(lambda x: delimiter.join(x), axis=1)
    var_split['Variable']=var_split['Variable'].str.rstrip(delimiter)    
    return (var_split['Variable'])

