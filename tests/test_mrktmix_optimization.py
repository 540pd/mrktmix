import mrktmix as mmm


def test_optimize_spend():
    # with out lower bound and upper bound
    optimum_sol = mmm.optimize_spend(10000, [85.0, 58.0, 70.0], [0.6, 0.4, 0.8], constraint='spend', verbosity=0)
    actual = list(*[*optimum_sol["freevariables"].values()])
    expected = [1956.953269488499, 403.28157861028035, 7639.765235633719]
    assert all([a == b for a, b in zip(actual, expected)])

    # with lower bound and upper bound
    optimum_sol = mmm.optimize_spend(
        10000,
        [85.0, 58.0, 70.0],
        [0.6, 0.4, 0.8],
        spend_lower_limit=[10, 500, 1],
        spend_upper_limit=[10000, 500000, 1000],
        revenue_lower_limit=[500, 696.65, 70.00],
        revenue_upper_limit=[21351.03, 11041.19, 17583.21],
        constraint='spend', verbosity=0)
    actual = list(*[*optimum_sol["freevariables"].values()])
    expected = [7839.090441464894, 1160.9084481323118, 999.9999998719591]
    assert all([a == b for a, b in zip(actual, expected)])
