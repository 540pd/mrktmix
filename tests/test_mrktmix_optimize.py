from pytest import approx

import mrktmix.optimization as optim


def test_optmization():
    coef = [47, 75, 13, 63, 96, 25, 17]
    intial_spend = [806, 332, 173, 661, 286, 253, 978]
    power = [0.9, 0.32, 0.97, 0.53, 0.02, 0.86, 0.67]
    lb = [644.8, 265.6, 138.4, 528.8, 228.8, 202.4, 782.4]
    ub = [967.2, 398.4, 207.6, 793.2, 343.2, 303.6, 1173.6]
    optim_result = optim.optimization(coef,
                                      intial_spend,
                                      power,
                                      lb,
                                      ub,
                                      3489,
                                      contraint_type='budget')
    optimized_spend = [967.19999999, 265.6, 207.6, 733.8, 228.8, 303.6, 782.4]
    assert optim_result.x == approx(optimized_spend)
    assert optim_result.success

    optim_result = optim.optimization(coef,
                                      intial_spend,
                                      power,
                                      lb,
                                      ub,
                                      28511.75,
                                      contraint_type='revenue')
    optimized_spend = [869.72435105, 265.6, 138.4, 528.8, 228.8, 202.4, 782.4]
    assert optim_result.x == approx(optimized_spend)
    assert optim_result.success
