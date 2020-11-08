import numpy as np
import pandas as pd

from mrktmix.dataprep import transform as dp


def create_base(variable, date_input, freq, increasing=False, negative=False, periods=1, panel=None):
    """ Create dummy/base variable for modeling

    :param variable: Name of variables
    :type variable: list
    :param date_input: List of string, list or tuple of Dates used in base variable. If input type is str, periods and freq will be used
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
    if isinstance(variable, (np.ndarray, list, pd.Series)):
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


def aggregate_data(mdl_data, panel_agg={}, variable_agg={}, metric_index_var=[-1], metric_mean_code=[], delimeter="_"):
    """
    Aggregate modeling data based on panel level or variable level. By defult, sum is used for aggregation.If mean function
    needs to be applied in modeling data at given code/level in variable,  metric index variable and metric mean code must
    to be supplied. The function can only one value being mapped to aggregate panel or aggregate variable. metric mean code
    is matched after variable aggregation

    :param mdl_data: modeling data with panel level and columns. Panel must be represented as multiindex rows.
    :type mdl_data: pandas.DataFrame
    :param panel_agg: Information on panel level aggregation. Keys of dicationary represents new panel name and values represents
    name of panel to be aggregated. Default value is empty dictionary.
    :type panel_agg: dictionary with list values
    :param variable_agg: Information on variable level aggregation. Keys of dicationary represents new variable and values represents
    name of variables to be aggregated. Default value is empty dictionary.
    :type variable_agg: dictionary with list values
    :param metric_index_var: index for metric type in variable of modeling data after spliting variable by delimeter supplied.
    Default value is [-1]
    :type metric_index_var: list of integer
    :param metric_mean_code: list of metric where mean will be applied. mean code is searched before applying variable aggregation
    :type metric_mean_code: list of string
    :param delimeter: delimeter used in variable in modeling data
    :type delimeter: string
    :return: modeling dataframe after aggregation at panel and variable level
    :rtype: pandas.DataFrame
    """

    # check if one variable is mapped to multiple aggregate panel
    mapped_panels = [j for i in [*panel_agg.values()] for j in i]
    multiple_mapped_panels = [i for i in mapped_panels if mapped_panels.count(i) > 1]
    # check if one variable is mapped to multiple aggregate panel
    mapped_vars = [j for i in [*variable_agg.values()] for j in i]
    multiple_mapped_vars = [i for i in mapped_vars if mapped_vars.count(i) > 1]
    if len(multiple_mapped_panels) and len(multiple_mapped_vars):
        raise Exception('Panel and variable mapping contains duplicate values')
    elif (len(multiple_mapped_panels)):
        raise Exception('Panel mapping contains duplicate values')
    elif (len(multiple_mapped_vars)):
        raise Exception('Variable mapping contains duplicate values')

    # Reverse key and value of variable agg dictionary
    variable_agg_reverse = {}
    _ = [variable_agg_reverse.update({val_item: key}) for key, val in variable_agg.items() for val_item in val]
    # Reverse key and value of variable panel dictionary
    panel_agg_reverse = {}
    _ = [panel_agg_reverse.update({val_item: key}) for key, val in panel_agg.items() for val_item in val]
    # find variables for mean summary
    mdl_data_renamed = mdl_data.rename(index=panel_agg_reverse, columns=variable_agg_reverse)
    mean_vars = dp.parse_variable(pd.Series(mdl_data_renamed.columns, index=mdl_data_renamed.columns),
                                  metric_index_var,
                                  delimiter=delimeter,
                                  anti=False).isin(metric_mean_code)
    # sum aggregation
    sum_agg = (mdl_data_renamed.loc[:, [*~np.isin(np.array(mdl_data_renamed.columns), np.array([*mean_vars[mean_vars.values].index]))]]
               .sum(level=list(range(mdl_data_renamed.index.nlevels)))
               .sum(axis=1, level=0))
    # mean aggregation
    if len([*mean_vars[mean_vars.values].index]):
        mean_agg = (mdl_data_renamed[[*mean_vars[mean_vars.values].index]]
                    .mean(level=mdl_data_renamed.index.names)
                    .mean(axis=1, level=0))
        agg_mdl_data = pd.concat([sum_agg, mean_agg], axis=1)
    else:
        agg_mdl_data = sum_agg
    return(agg_mdl_data)


def segregate_data(aggregated_data, segregated_data, panel_agg={}, variable_agg={}, panel_agg_index=0, match_sum=True):
    """
    Segregate aggregated data based on ratio in segregated data. Aggregated data can have panel level aggregation or variable
    level aggregation. If panel level aggregation is present, its mapping must be present in panel_agg. If variable level
    aggregation present, it must be present in variable_agg. Panel level aggregation is applied on panel_agg_index level
    of row index.

    :param aggregated_data: Aggregated data at panel level
    :type aggregated_data: pandas.DataFrane
    :param segregated_data: Segregated data is used to compute proportion for segregation of aggregated data
    :type segregated_data: pandas.DataFrame
    :param panel_agg: Information on panel level aggregation. Keys of dicationary represents new panel name (in aggregated data)
    and values represents name of panel (in segregated data) to be aggregated. Default value is empty dictionary.
    :type panel_agg: dictionary with list values
    :param variable_agg: Information on variable level aggregation. Keys of dicationary represents new variable (in aggregated data)
    and values represents name of variables (in segregated data) to be aggregated. Default value is empty dictionary.
    :type variable_agg: dictionary with list values
    :param match_sum: When proportion calculated in segregated data is 0 and corresponding value is present in aggregated data,
    then sum of aggregated data won't match segregated output data. If match_sum is True, values in segregated output data is
    adjusted to match its sum with sum of aggregated data
    :type match_sum: bool
    :return: segregated data at panel level
    :rtype: pandas.DataFrame
    """

    # Subset relevent variable_agg_subset
    variable_agg_subset = {}
    _ = [variable_agg_subset.update({i: j}) for i, j in variable_agg.items() if i in aggregated_data.columns]
    # Subset relevent panel_agg_subset
    panel_agg_subset = {}
    _ = [panel_agg_subset.update({i: j}) for i, j in panel_agg.items(
    ) if i in aggregated_data.index.get_level_values(panel_agg_index).unique()]

    # aggregate segregated data at panel level
    panel_level_seg_agg = (aggregate_data(segregated_data, panel_agg=panel_agg_subset, variable_agg={})
                           .reindex(aggregated_data.index))
    # Variable level segregation on panel level aggregated data
    var_level_segg = pd.DataFrame()
    for agg_var, seg_vars in variable_agg_subset.items():
        var_level_segg_temp = dp.segregate_variable(aggregated_data[agg_var], panel_level_seg_agg[seg_vars], match_sum=match_sum)
        var_level_segg = pd.concat([var_level_segg, var_level_segg_temp], axis=1)
    var_level_segg_all = pd.concat(
        [var_level_segg, aggregated_data.loc[:, aggregated_data.columns.isin(panel_level_seg_agg.columns)]], axis=1)
    # Panel level segregation on variable level segregated data
    panel_level_segg = pd.DataFrame()
    for agg_panel, seg_panel in panel_agg_subset.items():
        panel_level_segg_temp = segregated_data.loc[segregated_data.index.get_level_values(
            panel_agg_index).isin(seg_panel), var_level_segg_all.columns]
        panel_level_agg = (var_level_segg_all[var_level_segg_all.index.get_level_values(panel_agg_index).isin([agg_panel])]
                           .droplevel(level=panel_agg_index))
        panel_level_segg = pd.concat([panel_level_segg, dp.segregate_panel(
            panel_level_agg, panel_level_segg_temp, match_sum=match_sum)], axis=0)
    panel_var_seg_data = pd.concat(
        [panel_level_segg,
         var_level_segg_all[~var_level_segg_all.index.get_level_values(panel_agg_index).isin([*panel_agg.keys()])]], axis=0)
    return(panel_var_seg_data)


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


def apply_coef_(raw_data, coef, dep_series):
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
    if coef.index.nlevels != raw_data.index.nlevels:
        raise Exception('Mismatch of index in input data')
    if coef.index.nlevels - 1:
        panel_var = (pd.Series(coef.index.get_level_values(-1))
                     .groupby(coef.droplevel(-1).index)
                     .apply(list)
                     .to_dict())
    else:
        panel_var = {False: [*coef.index]}

    after_apl = apply_apl(raw_data, panel_var)
    if coef.index.nlevels - 1:
        coef2frame = coef.unstack()
        coef2frame.columns = pd.MultiIndex.from_tuples(coef2frame.columns)
        dep_decomposition = after_apl[coef2frame.columns].mul(coef2frame.reindex(after_apl.index, level=0))
    else:
        dep_decomposition = after_apl.mul(coef)
    if dep_series is not None:
        dep_decomposition[("Residual", 0, 1, 0)] = dep_series - dep_decomposition.sum(axis=1)
    return(dep_decomposition)


def apply_coef_node_(mdl_data, coef, nodes, node, node_split):
    """
    Apply coef on modeling data to create decomposition.

    :param mdl_data: modeling dataframe with date index at level -1. If panel is present, it should be at level -2.
    :type mdl_data: pandas.DataFrame
    :param coef: Coefficient to be applied on modeling dataframe. Coefficient should have tuple of variable, adstock,power
        and lag at index level -1. If panel is present in modeling data, then coefficient must have panel information in index at level -2.
    :type coef: pandas.Series
    :param nodes: Relationship between indepenent variable and node to be applied on modeling dataframe. Nodes should
    have tuple of independent variable, adstock,power and lag at index level -1. If panel is present in modeling data, then nodes
    must have panel information in index at level -2. Dependent variables should be as value of series
    :type nodes: pandas.Series
    :param node: Selected nodes for which decomposition will be created
    :type dep_series: string
    :param node_split: Total number of relationshipship for the given node. It will equally seperate decomposition in-between the nodes
    :type node_split: int
    :return: Decomposition of dependent series for given node
    :rtype: pandas.DataFrame
    """

    # apply apl and create model decomposition
    df_apl = apply_coef_(
        mdl_data,
        coef[nodes == node],
        apply_apl(mdl_data, {False: [node]}).iloc[:, 0])
    # Decomposition of nodes
    if node_split:
        adj_decomposition = pd.concat([
            df_apl.loc[:, df_apl.columns.get_level_values(0).isin(["Intercept", "Residual"])].sum(axis=1).rename(node),
            (df_apl.assign(_Adjustment_Factor_=lambda x: x.sum(axis=1) * -1)
             .drop(columns=["Intercept", "Residual"], level=0))], axis=1)
    else:
        adj_decomposition = df_apl
    dep_decomposition = (pd.concat([adj_decomposition], axis=1, keys=[("", node)])
                         .droplevel(level=0, axis=1))
    return(dep_decomposition)


def apply_coef(mdl_data, coef, nodes=None, dep_series=None):
    """
    Apply coef on modeling data to create decomposition. If given node is present more than one relationships, then decomposed
    series is equally divided into the independent nodes.

    :param mdl_data: modeling dataframe with date index at level -1. If panel is present, it should be at level -2.
    :type mdl_data: pandas.DataFrame
    :param coef: Coefficient to be applied on modeling dataframe. Coefficient should have tuple of variable, adstock,power
        and lag at index level -1. If panel is present in modeling data, then coefficient must have panel information in index at level -2.
    :type coef: pandas.Series
    :param nodes: Relationship between indepenent variable and node to be applied on modeling dataframe. Nodes should
    have tuple of independent variable, adstock,power and lag at index level -1. If panel is present in modeling data, then nodes
    must have panel information in index at level -2. Dependent variables should be as value of series. Default value is None
    :type nodes: pandas.Series
    :param dep_series: Dependent Series will be used to calculate residual. It must be at same level modeling dataframe. Default
    is None. If dependent is not None, then residuals will also be calculated
    :type dep_series: pandas.Series or None
    :return: Decomposition of dependent series for given node
    :rtype: pandas.DataFrame
    """
    if nodes is None:
        # no nodes are present. simple case of decomposition
        return(apply_coef_(mdl_data, coef, dep_series))
    else:
        # Panel is present
        if nodes.index.nlevels == 2:
            network_decomposition = pd.DataFrame()
            for panel in nodes.index.get_level_values(0).unique():
                # count of nodes
                nodes_count = {node: sum(coef[panel][nodes[panel] != node].index == node) for node in nodes[panel].unique()}
                panel_decomposition = pd.concat([
                    apply_coef_node_(mdl_data.loc[[(panel)], :],
                                     coef[[panel]],
                                     nodes[[panel]],
                                     node,
                                     node_split) for node, node_split in nodes_count.items()], axis=1)
                network_decomposition = pd.concat([network_decomposition, panel_decomposition])
        # Panel is not present
        else:
            # count of nodes
            nodes_count = {node: sum(coef[nodes != node].index == node) for node in nodes.unique()}
            network_decomposition = pd.concat([apply_coef_node_(mdl_data, coef, nodes, node, node_split)
                                               for node, node_split in nodes_count.items()], axis=1)
        return(network_decomposition)


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
        if dep_decompose.index.nlevels - 1:
            decomp_smry = dep_decompose.iloc[selected_date].sum(level=range(0, dep_decompose.index.nlevels - 1))
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


def assess_error(dep_decompose):
    """ Create actual vs predicted with error terms from given resonse decomposition

    :param dep_decompose: data with decomposition of response variable and Residual term
    :type dep_decompose: pandas.DataFrame
    :return: dataframe with actual, predicted, error and error % to measure accury of model
    :rtype: pandas.DataFrame
    """
    dep = dep_decompose.sum(axis=1).rename("Dependent")
    pred = dep_decompose.drop("Residual", axis=1, level=-4).sum(axis=1).rename("Predicted")
    residual = (dep - pred).rename("Error")
    residual_perc = (residual / dep).rename("Error %")
    return(pd.concat([dep, pred, residual, residual_perc], axis=1))
