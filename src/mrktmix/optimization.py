import numpy as np
from gpkit import Model
from gpkit import SignomialsEnabled
from gpkit import Variable
from gpkit import VectorVariable


def optimize_spend(
        constraint_ceiling,
        coefficient,
        exponent,
        spend_lower_limit=None,
        spend_upper_limit=None,
        revenue_lower_limit=None,
        revenue_upper_limit=None,
        constraint="spend",
        verbosity=0):
    """Optimizes spend based on constraint_ceiling, coefficient and power. constaint_ceiling can be based on spend or revenue which
    can be specified in constraint parameter. If constraint is spend, then optimization will be spend based. It will use total spend
    as budget and tries to maximize total revenue. If constraint is revenue, then optimization will be goal based. It will use total
    revenue and tries to minimize total spend to achieve target revenue. One can also include additional parameters like lower bound
    and upper bound for spend and revenue for optimization.
    Optimization formulation: Total revenue = sum of all {coefficient*spend^(exponent)}. Each coeffient, spend and exponent is assumed
    for every marketing hiearchy.
    Assumption: Coefficient needs to be positive

    :param constraint_ceiling: ceiling or contrains for spend or revenue used for optimization
    :type constraint_ceiling: numeric
    :param coefficient: coeffient from modeling for given marketing hiearchy
    :type coefficient: list of numeric values
    :param exponent: exponent or power for given marketing hiearchy
    :type exponent: list of numeric values
    :param spend_lower_limit: lower bound for spend for optimization
    :type spend_lower_limit: None or list of numeric values
    :param spend_upper_limit: upper bound for spend for optimization
    :type spend_upper_limit: None or list of numeric values
    :param revenue_lower_limit: lower bound for revenue for optimization
    :type revenue_lower_limit: None or list of numeric values
    :param revenue_upper_limit: upper bound for revenue for optimization
    :type revenue_upper_limit: None or list of numeric values
    :param constraint: attribute for ceiling or contrains for optimization. Its value will be spend if objective is spend based
    optimization or value will be revenue if the goal is goal based optimization. By default, its value is spend.
    :type constraint: string
    :param verbosity: Can be used to print details of optimization. By default its value is 0, which will print short description
    of optimization.
    :type verbosity: 0 or 1
    :return: optimized object from gpkit
    :rtype: gpkit.solution_array.SolutionArray
    """

    no_of_optim_params = len(coefficient)
    # convert to coefficent and power to float
    coef = [float(item) for item in coefficient]
    power = [float(item) for item in exponent]
    if spend_lower_limit is None:
        spend_lower_limit = [0] * no_of_optim_params
    if spend_upper_limit is None:
        spend_upper_limit = [float("Inf")] * no_of_optim_params
    if revenue_lower_limit is None:
        revenue_lower_limit = [0] * no_of_optim_params
    if revenue_upper_limit is None:
        revenue_upper_limit = [float("Inf")] * no_of_optim_params

    # Parameters
    if constraint.lower() == 'revenue':
        b = Variable("b", constraint_ceiling, "-", "total revenue")
    else:
        b = Variable("b", constraint_ceiling, "-", "budget")
    s_min = VectorVariable(no_of_optim_params, "s_{min}", spend_lower_limit, "-", "lower bound for spend")
    s_max = VectorVariable(no_of_optim_params, "s_{max}", spend_upper_limit, "-", "upper bound for spend")
    r_min = VectorVariable(no_of_optim_params, "r_{min}", revenue_lower_limit, "-", "lower bound for revenue")
    r_max = VectorVariable(no_of_optim_params, "r_{max}", revenue_upper_limit, "-", "upper bound for revenue")

    # Decision Variable
    s = VectorVariable(no_of_optim_params, "s", "-", "spend")
    r = VectorVariable(no_of_optim_params, "r", "-", "revenue")
    r = coef * np.power(s, power)
    r_inv = VectorVariable(no_of_optim_params, "r_inv", "-", "inverse of individual revenue")
    # r_inv = np.power(coef, -1) * np.power(s, np.multiply(-1, np.power(power, -1)))
    r_inv = coef * np.power(s, np.multiply(-1, power))

    # Constraints
    # must enable signomials for subtraction
    if constraint.lower() == 'revenue':
        # goal/revenue based optimization
        with SignomialsEnabled():
            constraints = [s >= s_min, s <= s_max, r >= r_min, r <= r_max, sum(r) >= b]
    else:
        # spend based optimization
        constraints = [s >= s_min, s <= s_max, r >= r_min, r <= r_max, sum(s) <= b]
    # Objective function
    if constraint.lower() == 'revenue':
        # goal/revenue based optimization
        objective = sum(s)
    else:
        # spend based optimization
        objective = sum(r_inv)

    # Formulate the Model
    m = Model(objective, constraints)
    # Solve the Model and print the results table
    if constraint.lower() == 'revenue':
        sol = m.localsolve(verbosity=verbosity)
    else:
        sol = m.solve(verbosity=verbosity)
    return(sol)
