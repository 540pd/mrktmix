import numpy as np
import pandas as pd
from scipy.optimize import Bounds
from scipy.optimize import minimize


def optimization(_coef,
                 _init,
                 _power,
                 lower_bound,
                 upper_bound,
                 tot_constraint,
                 contraint_type="budget"):
    """ Optimize revenue/budget (Revenue= summation of coef * spend ^ power) based in initial value and constraint. Constraints
    is in form of lower bound, upper bound and either budget or revenue

    :param _coef: list of float representing coefficients
    :type _coef: list of float
    :param _init: list of float representing initial spend
    :type _init: list of float
    :param _power: list of float representing diminshing return on spend
    :type _power: list of float
    :param lower_bound: list of number representing lower bound for spend
    :type lower_bound: list of number
    :param upper_bound: list of number representing upper bound for spend
    :type upper_bound: list of number
    :param tot_constraint: total contraint either total spend or total revenue
    :type tot_constraint: numeric
    :param contraint_type: defines type for tot_constraint. Default value is 'budget'
    :type contraint_type: string, optional
    :return: Optimized result
    :rtype: scipy.optimize.optimize.OptimizeResult
    """

    _coef = pd.Series(_coef)
    _power = pd.Series(_power)
    _init = pd.Series(_init)

    if contraint_type == "revenue":
        def objective(x):
            return np.sum(x)

        def obj_der(x):
            return pd.Series([1] * len(x))

        def constrain(x, tot_constraint=tot_constraint, _coef=_coef, _power=_power):
            return tot_constraint - np.sum(_coef * (x**_power))
    else:
        def objective(x, _coef=_coef, _power=_power):
            return -np.sum(_coef * (x**_power))

        def obj_der(x, _coef=_coef, _power=_power):
            return -_coef * _power * (x**(_power - 1))

        def constrain(x, tot_constraint=tot_constraint):
            return tot_constraint - sum(x)
    return (minimize(fun=objective, x0=_init, jac=obj_der, method='SLSQP',  # trust-constr,SLSQP
                     bounds=Bounds(lower_bound, upper_bound),
                     constraints=[{'type': 'eq', 'fun': constrain}]))
