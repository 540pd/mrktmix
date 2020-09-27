===========
Quick Start
===========

Assuming you have Python already, install mrktmix::

    import pandas as pd
    import numpy as np
    import mrktmix as mmm

While building models, one often have to create baseline which represents unexplained or exogenous part of response variable. To crease base for model::

    # Create base for dates specified i.e. date input is list
    mmm.create_base("var1", ["2020-03-10", "2020-04-07"], "7D", increasing=False, negative=False, panel=None))
	
    # Create base for given range of dates i.e. date input is tuple
    mmm.create_base("var1", ("2020-03-10", "2020-03-21"), "7D", increasing=False, negative=False, panel=None))
	
    # Create base from given date i.e. date input is string
    mmm.create_base("var1", "2020-03-10", "7D", increasing=False, negative=False, periods=3, panel=None))
	
    # Create base from pandas series
    df = pd.DataFrame([["var2", ["2020-03-10", "2020-04-07"]], ["var2", ("2020-03-10", "2020-04-07")], ["var3", ("2020-03-10", "2020-04-06")], ["var4", "2020-03-10"]])
    mmm.create_base(df[0].values, df[1].values, "7D", increasing=False, negative=False, periods=5, panel=["panel1", "panel1", "panel2", "panel1"]))

To apply adstock, power and lag transformation, input data must have sorted date in level -1 of row multiindex. Additional hiearachy like panel can be present in row multiindex. If there is only date index in data, key of dictionary input must be False. If additional hiearchy is present in data like panel, then key of dictionary input can be name of panels if transformation is applicable to selected panel or True if tranformation is applicable to all the panels present in data::

    # Apply adstock, power and lag transformation when there is no panel in input data i.e. row index have date index only
    variables = {False: [('Intercept', 0, 1, 0), ('one', 0, 1, 0), ('two', 0, 1, 0), ('one', 0, 1, 1)]}
    df_1 = pd.DataFrame({'two': [1., 2., 3., 4.], 'one': [4., 3., 2., 1.], 'Intercept': [4., 3., 2., 1.]}, index=['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'])
    mmm.apply_apl(df_1, variables)


Apply modeling/regression coefficient and create decomposition of response variable::

    # Create model coefficient
    ind = [np.repeat(["METRO","CITY"], 2), [("Intercept", 0, 1, 0), ("A", 0, 1, 0), ("Intercept", 0, 1, 0), ("B", 0, 1, 0)]]
    coef= pd.Series(data=[3240, 600, 10, 0.9], index=ind)
    # Create data
    df_ind = [np.repeat(["METRO","CITY"], 3), np.tile(pd.date_range(start='1/1/2018', end='1/03/2018'),2)]
    df = pd.DataFrame({'A': [773, 137, 508, 562, 365, 500], 'B': [848, 326, 969, 730, 761, 137]}, index=df_ind)
    df["Intercept"] = 1
    # Create response data
    dep = pd.DataFrame({'Dep': [773, 137, 508, 562, 365, 500]}, index=df_ind)
    # Apply coefficient
    mmm.apply_coef(df, coef, dep["Dep"])

Summarize decomposition of response variable::

    dep_decompose=apply_coef(df, coef, dep["Dep"])
    date_dict = {'test Year': ('3/1/2018', '3/1/2018'), 'train_year': ('1/1/2018', '2/1/2018')}
    mmm.collapse_date(dep_decompose, date_dict)

Compare response series and predicted series along with errors from decomposition of response variable::

    mmm.assess_error(apply_coef(df, coef, dep["Dep"]))
