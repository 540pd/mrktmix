=====
Usage
=====

To use Market Mix Modeling in a project::

	import mrktmix as mmm

Example::

    import pandas as pd
    import numpy as np

    # Apply adstock, power and lag transformation on pandas series
    dates = (pd.date_range(start='20180601', end='20180630', freq='W-FRI').strftime('%Y-%m-%d'))
    df = pd.Series(np.array([1., 2., 3., 4., 5.]), index=dates, name="two")
    mmm.apply_apl(df, 0.5, .7, 1)
		
    # Create base for dates specified i.e. date input is list
    mmm.create_base("var1", ["2020-03-10", "2020-04-07"], "7D", increasing=False, negative=False, panel=None))
	
    # Create base for given range of dates i.e. date input is tuple
    mmm.create_base("var1", ("2020-03-10", "2020-03-21"), "7D", increasing=False, negative=False, panel=None))
	
    # Create base from given date i.e. date input is string
    mmm.create_base("var1", "2020-03-10", "7D", increasing=False, negative=False, periods=3, panel=None))
	
    # Create base from pandas series
    df = pd.DataFrame([["var2", ["2020-03-10", "2020-04-07"]], ["var2", ("2020-03-10", "2020-04-07")], ["var3", ("2020-03-10", "2020-04-06")], ["var4", "2020-03-10"]])
    mmm.create_base(df[0].values, df[1].values, "7D", increasing=False, negative=False, periods=5, panel=["panel1", "panel1", "panel2", "panel1"]))

