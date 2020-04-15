import numpy as np
from scipy.optimize import fmin


def func_att_diff(x, x_target):
    """ gives the difference between the current flux and the asked one

    @param[in] x             current value of the attenuator
    @param[in] x_target      fraction of the flux you are looking for

    @returns                 difference of flux
    """

    a = 1
    b = -0.337
    c = -0.159
    d = 0.054
    if x < 255:
        return np.abs(a + b * (x / 100) + c * (x / 100) ** 2 + d * (x / 100) ** 3 - x_target)
    else:
        return np.abs(a - x_target)


def func_att(x):
    """ gives the fraction of flux given the attenuator value

    @param[in] x             current value of the attenuator

    @returns                 fraction of the full flux
    """
    a = 1
    b = -0.337
    c = -0.159
    d = 0.054
    return a + b * (x / 100) + c * (x / 100) ** 2 + d * (x / 100) ** 3


# list of defocus_values, movement of the slit
list_of_defocus_exposure_times = np.array([-4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1., -0.5, 0., 0.5,
                                           1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5])

# polynomial that gives exp time as a function of defocus
# p2 = np.poly1d([2.67750812, -0.22915717, 1.])
p2 = np.poly1d([0.03958005, 5.26200135, -1.24528564, 1])
# helper value, when looking for best attenuator value start search  x = 0
x0 = 0


#
# value_att = fmin(lamda_f,x0)


def defocused_exposure_times(exp_time_0, att_value_0):
    """ gives list of exposure time and attenatuor values for defocused experiment

    @param[in] exp_time_0            exposure time for focus
    @param[in] att_value_0           attenuator value  for focus

    @returns                        array of exposure times, array of attenuator values
    """
    defocused_exposure_times = []
    att_values = []

    if att_value_0 == 0:
        effective_exp_time_0 = exp_time_0
    else:
        effective_exp_time_0 = exp_time_0 * (func_att(att_value_0))

    for i in list_of_defocus_exposure_times:
        time_without_att = p2(i) * effective_exp_time_0
        if time_without_att < 15:
            how_much_too_bright = 15 / time_without_att

            lamda_f = lambda x: func_att_diff(x, 1 / how_much_too_bright)

            value_att = fmin(lamda_f, 0)
            time_without_att = 15
            att_values.append(value_att[0])
        else:
            att_values.append(0)
        defocused_exposure_times.append(time_without_att)

    return (np.round(defocused_exposure_times).astype(int), np.round(att_values).astype(int))


def defocused_exposure_times_single_position(exp_time_0, att_value_0, defocused_value):
    """ gives list of exposure time and attenatuor values for defocused experiment

    @param[in] exp_time_0            exposure time for focus
    @param[in] att_value_0           attenuator value  for focus
    @param[in] defocused_value       value for the slit positions

    @returns                        single exposure time, single attenuator value
    """

    defocused_exposure_times = []
    att_values = []

    if att_value_0 == 0:
        effective_exp_time_0 = exp_time_0
    else:
        effective_exp_time_0 = exp_time_0 * (func_att(att_value_0))

    i = defocused_value
    time_without_att = p2(i) * effective_exp_time_0

    if time_without_att < 15:
        how_much_too_bright = 15 / time_without_att

        lamda_f = lambda x: func_att_diff(x, 1 / how_much_too_bright)

        value_att = fmin(lamda_f, 0)
        time_without_att = 15
        att_values.append(value_att[0])
    else:
        att_values.append(0)
    defocused_exposure_times.append(time_without_att)
    return np.round(defocused_exposure_times[0]).astype(int), np.round(att_values[0]).astype(int)
