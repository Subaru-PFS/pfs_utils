#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import numpy as np


class OptimizeBlackDots(object):
    """Optimization of position of black dots

    Class with functions for modifying the positions of black dots
    for PFS spectrograph

    Parameters
    ----------
    dots : pf.dataframe
        dataframe contaning initial guesses for dots
    list_of_mcs_data_all : `list` of `np.arrays`
        list contaning mcs observations
    list_of_descriptions : `list` of `str`
        what kind of moves have been made (theta or phi)
    """

    def __init__(self, dots, list_of_mcs_data_all, list_of_descriptions):

        self.dots = dots
        self.n_of_obs = len(list_of_descriptions)

        # example for November 2021 run
        # list_of_mcs_data_all = [mcs_data_all_1, mcs_data_all_2]
        # list_of_descriptions = ['theta', 'phi']
        list_of_prepared_observations_and_predictions = []
        for obs in range(len(list_of_mcs_data_all)):
            mcs_data_all = list_of_mcs_data_all[obs]
            description = list_of_descriptions[obs]
            prepared_observations_and_predictions =\
                self.predict_positions(mcs_data_all, description)
            list_of_prepared_observations_and_predictions.append(prepared_observations_and_predictions)
        self.list_of_prepared_observations_and_predictions = list_of_prepared_observations_and_predictions

    def return_list_of_prepared_observations_and_predictions(self):
        """Returns the observed and predicted positions
        """
        return self.list_of_prepared_observations_and_predictions

    def predict_positions(self, mcs_data_all, type_of_run='theta'):
        """Predict positions of spots which were not observed

        Parameters
        ----------
        mcs_data_all: `np.array`
           positions of observed spots
         type_of_run: `str`
             describing which kind of movement is being done
             (theta or phi)

        Returns
        ----------
        prepared_observations_and_predictions `list`
            lists contaning 4 arrays, which contanin observed position in x and y,
            and predicted position in x and y
        """

        # lists which will contain the positions which have been observed
        # 2 runs and 2 coordinates = 4 lists
        list_of_actual_position_x_observed = []
        list_of_actual_position_y_observed = []

        # lists which will contain the position which were not observed
        # and we predicted with this algorithm
        list_of_predicted_position_x_not_observed = []
        list_of_predicted_position_y_not_observed = []

        for i in range(0, 2394):
            try:
                x_measurments = mcs_data_all[0, i]
                y_measurments = mcs_data_all[1, i]

                # index of the points which are not observed
                idx = np.isfinite(x_measurments) & np.isfinite(y_measurments)

                # select the point that are more than 2 mm away from median of all of the points
                # replace them with np.nan
                # these points have been attributed to wrong cobras
                # ignore warning if idx_1 is all False, i.e., you didnt observe any of the points
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    large_outliers_selection = (np.abs(y_measurments -
                                                       np.median(y_measurments[idx])) > 2) |\
                        (np.abs(x_measurments - np.median(x_measurments[idx])) > 2)
                y_measurments[large_outliers_selection] = np.nan
                x_measurments[large_outliers_selection] = np.nan

                # if np.sum(idx_1)<10:
                #    raise ValueError('too few points')

                # create prediction for where the points should be, when you do not see points
                # if you do not see points from both sides of the black dots, do simpler extrapolation
                # (np.sum(np.diff(idx_1)) == 1)
                # if you see points from both sides of the black dots, do more complex interpolation
                # (np.sum(np.diff(idx_1)) != 1)
                if type_of_run == 'theta':
                    if np.sum(np.diff(idx)) == 1:
                        deg2_fit_to_single_point_y = np.polyfit(np.arange(0, len(y_measurments))[idx],
                                                                y_measurments[idx], 3)
                        deg2_fit_to_single_point_x = np.polyfit(np.arange(0, len(y_measurments))[idx],
                                                                x_measurments[idx], 3)
                    else:
                        deg2_fit_to_single_point_y = np.polyfit(np.arange(0, len(y_measurments))[idx],
                                                                y_measurments[idx], 4)
                        deg2_fit_to_single_point_x = np.polyfit(np.arange(0, len(y_measurments))[idx],
                                                                x_measurments[idx], 4)
                else:
                    if np.sum(np.diff(idx)) == 1:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            deg2_fit_to_single_point_y, residuals_y =\
                                np.polyfit(np.arange(0, len(y_measurments))[idx],
                                           y_measurments[idx], 1, full=True)[0:2]
                            deg2_fit_to_single_point_x, residuals_x =\
                                np.polyfit(np.arange(0, len(x_measurments))[idx],
                                           x_measurments[idx], 1, full=True)[0:2]
                    else:
                        deg2_fit_to_single_point_y, residuals_y =\
                            np.polyfit(np.arange(0, len(y_measurments))[idx],
                                       y_measurments[idx], 3, full=True)[0:2]
                        deg2_fit_to_single_point_x, residuals_x =\
                            np.polyfit(np.arange(0, len(x_measurments))[idx],
                                       x_measurments[idx], 3, full=True)[0:2]

                    if len(residuals_y) == 0:
                        residuals_y = np.array([99])
                    if len(residuals_x) == 0:
                        residuals_x = np.array([99])

                poly_deg_x = np.poly1d(deg2_fit_to_single_point_x)
                poly_deg_y = np.poly1d(deg2_fit_to_single_point_y)

                # some cleaning, to exclude the results where the interpolation results make no sense
                if type_of_run == 'theta':
                    if np.sum(~idx) > 5:
                        r_distance_predicted = np.sqrt((poly_deg_x
                                                        (np.arange(0, len(x_measurments)))[~idx][-1] -
                                                        poly_deg_x
                                                        (np.arange(0, len(x_measurments)))[~idx][0])**2 +
                                                       (poly_deg_y
                                                        (np.arange(0, len(x_measurments)))[~idx][-1] -
                                                        poly_deg_y
                                                        (np.arange(0, len(x_measurments)))[~idx][0])**2)
                    else:
                        r_distance_predicted = 0
                else:
                    if np.sum(~idx) > 5:
                        r_distance_predicted = np.sqrt((poly_deg_x
                                                        (np.arange(0, len(x_measurments)))[~idx][-1] -
                                                        poly_deg_x
                                                        (np.arange(0, len(x_measurments)))[~idx][0])**2 +
                                                       (poly_deg_y
                                                        (np.arange(0, len(x_measurments)))[~idx][-1] -
                                                        poly_deg_y
                                                        (np.arange(0, len(x_measurments)))[~idx][0])**2)
                    else:
                        r_distance_predicted = 0

                if type_of_run == 'theta':
                    if (~(np.sum(np.diff(idx)) == 1 and r_distance_predicted > 2.5)):
                        predicted_position_x = poly_deg_x(np.arange(0, len(x_measurments)))
                        predicted_position_y = poly_deg_y(np.arange(0, len(y_measurments)))
                    else:
                        predicted_position_x = np.empty(len(x_measurments))
                        predicted_position_x[:] = np.NaN
                        predicted_position_y = np.empty(len(y_measurments))
                        predicted_position_y[:] = np.NaN
                else:
                    if residuals_y[0] < 0.5 and residuals_x[0] < 0.5\
                            and ~((np.sum(np.diff(idx)) == 1) and (r_distance_predicted > 1.5)):
                        predicted_position_x = poly_deg_x(np.arange(0, len(x_measurments)))
                        predicted_position_y = poly_deg_y(np.arange(0, len(y_measurments)))
                    else:
                        predicted_position_x = np.empty(len(x_measurments))
                        predicted_position_x[:] = np.NaN
                        predicted_position_y = np.empty(len(y_measurments))
                        predicted_position_y[:] = np.NaN

                # create lists which contain observed and predicted data
                actual_position_x_observed = mcs_data_all[0, i][idx]
                actual_position_y_observed = mcs_data_all[1, i][idx]

                predicted_position_x_not_observed = predicted_position_x[~idx]
                predicted_position_y_not_observed = predicted_position_y[~idx]

                list_of_actual_position_x_observed.append(actual_position_x_observed)
                list_of_actual_position_y_observed.append(actual_position_y_observed)

                list_of_predicted_position_x_not_observed.append(predicted_position_x_not_observed)
                list_of_predicted_position_y_not_observed.append(predicted_position_y_not_observed)
            except (TypeError, ValueError):
                # if the process failed at some points, do not consider this fiber in the later stages
                list_of_actual_position_x_observed.append(np.nan)
                list_of_actual_position_y_observed.append(np.nan)

                list_of_predicted_position_x_not_observed.append(np.nan)
                list_of_predicted_position_y_not_observed.append(np.nan)

        prepared_observations_and_predictions = [list_of_actual_position_x_observed,
                                                 list_of_actual_position_y_observed,
                                                 list_of_predicted_position_x_not_observed,
                                                 list_of_predicted_position_y_not_observed]

        return prepared_observations_and_predictions

    def rotate(self, point, origin, degrees):
        """rotate the point around the origin

        gets called by `new_position_of_dot`

        Parameters
        ----------
        point: `tuple`
            x, y coordinate of the point to rotate
        origin: `tuple`
            x, y coordinate of the point around which to rotate
        degrees: `float`
            number of degrees to rotate

        Returns
        ----------
        x, y coordinate of the rotated point
        """
        radians = np.deg2rad(degrees)
        x, y = point
        offset_x, offset_y = origin
        adjusted_x = (x - offset_x)
        adjusted_y = (y - offset_y)
        cos_rad = np.cos(radians)
        sin_rad = np.sin(radians)
        qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
        qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
        return qx, qy

    def check(self, x, y, xc, yc, r=None):
        """check if the cobra is covered by the black dot

        gets called by `quality_measure_single_spot`

        Parameters
        ----------
        x: `float`
            x coordinates of the dot
        y: `float`
            y coordinate of the dot
        xc: `float`
            x coordinate of the cobra
        yc: `float`
            y coordinate of the cobra
        r: `float`
            radius of dot [mm]

        Returns
        ----------
        `int`
         returns 1 if the cobra is covered, 0 if not
        """
        if r is None:
            r = 0.75
        return int(((x - xc)**2 + (y - yc)**2) < r**2)

    def new_position_of_dot(self, i, xd_mod, yd_mod, scale, rot, x_scale_rot, y_scale_rot):
        """move the dot to the new position

        This function defines the change of the positions that we consider

        calls `rotate`

        Parameters
        ----------
        i: `integer`
            index of the cobra
        xd_mod: `float`
            x offset of the new position
        yd_mod: `float`
            y offset of the new position
        scale: `float`
            overall scaling
        x_scale_rot: `float`
            x coordinate around which to rotate
        x_scale_rot: `float`
            y coordinate around which to rotate
        Returns
        ----------
        xd_new, yd_new: `float`, `float`
            x and y coordinate of the new position
        """
        dots = self.dots

        xd_original = dots.loc[i-1]['x']
        yd_original = dots.loc[i-1]['y']

        xd_offset = xd_original - x_scale_rot
        yd_offset = yd_original - y_scale_rot

        r_distance_from_center = np.sqrt(xd_offset**2 + yd_offset**2)

        xc_rotated, yc_rotated = self.rotate((xd_offset*(1+scale*r_distance_from_center/100),
                                              yd_offset*(1+scale*r_distance_from_center/100)), (0, 0), rot)

        xc_rotated = xc_rotated+x_scale_rot
        yc_rotated = yc_rotated+y_scale_rot
        # print(xc_rotated , yc_rotated)
        xd_new = xc_rotated + xd_mod
        yd_new = yc_rotated + yd_mod
        return xd_new, yd_new

    def quality_measure_single_spot(self, list_of_prepared_observations_and_predictions_single_fiber,
                                    xd, yd):
        """calculate a penalty given x and y dot positions

        calls `check`

        gets called by `total_penalty_for_single_dot`

        Parameters
        ----------
        list_of_prepared_observations_and_predictions: `list`
            list contaning observations for only one fiber
        xd: `float`
            x position of the dot
        yd: `float`
            y position of the dot

        Returns
        ----------
        quality_measure: `float`
            total penalty for this position of the dot
        """
        list_of_quality_measure_single_run = []
        for i in range(len(list_of_prepared_observations_and_predictions_single_fiber)):
            prepared_observations_and_predictions =\
                list_of_prepared_observations_and_predictions_single_fiber[i]

            actual_position_x_observed,\
                actual_position_y_observed,\
                predicted_position_x_not_observed,\
                predicted_position_y_not_observed = prepared_observations_and_predictions

            # the ones which were seen - add penalty if you should not have seen them
            quality_measure_observed = []
            for j in range(len(actual_position_x_observed)):
                quality_measure_observed.append(self.check(actual_position_x_observed[j],
                                                           actual_position_y_observed[j], xd, yd, 0.75))

            # the ones which should not be seen - add penalty if you saw them
            quality_measure_predicted = []
            for j in range(len(predicted_position_x_not_observed)):
                # if you were not able to predict position, no penalty
                if np.isnan(predicted_position_x_not_observed[j]):
                    val = 0
                else:
                    val = self.check(predicted_position_x_not_observed[j],
                                     predicted_position_y_not_observed[j], xd, yd, 0.75)
                    val = 1 - val
                quality_measure_predicted.append(val)

            quality_measure_single_run = np.sum(quality_measure_observed) + np.sum(quality_measure_predicted)
            list_of_quality_measure_single_run.append(quality_measure_single_run)

        sum_quality_measure_single_run = np.sum(list_of_quality_measure_single_run)

        return sum_quality_measure_single_run

    def total_penalty_for_single_dot(self, i, xd, yd, print_outputs=False):
        """calculate a penalty for a single dot

        calls `quality_measure_single_spot`

        Parameters
        ----------
        i: `integer`
            index of the cobra
        xd: `float`
            x position of the dot
        yd: `float`
            y position of the dot
        print_outputs: `bool`
            if True print outputs while computing

        Returns
        ----------
        total_penalty_for_single_modification: `float`
            total penalty for this position of the dot
        """

        list_of_prepared_observations_and_predictions_single_fiber = []
        for obs in range(self.n_of_obs):
            list_of_prepared_observations_and_predictions_single_fiber_single_run = []
            for j in range(4):
                list_of_prepared_observations_and_predictions_single_fiber_single_run.append(
                    self.list_of_prepared_observations_and_predictions[obs][j][i])

            list_of_prepared_observations_and_predictions_single_fiber.append(
                list_of_prepared_observations_and_predictions_single_fiber_single_run)

        sigle_run_result_for_modification = []
        try:
            sigle_run_result_for_modification.\
                append([i, self.quality_measure_single_spot(
                    list_of_prepared_observations_and_predictions_single_fiber, xd, yd)])
        except (TypeError, ValueError):
            pass

        sigle_run_result_for_modification = np.array(sigle_run_result_for_modification)
        if print_outputs is True:
            print(sigle_run_result_for_modification)
        total_penalty_for_single_modification = np.sum(sigle_run_result_for_modification[:, 1])
        return total_penalty_for_single_modification

    def optimize_function(self, design_variables, return_full_result=False):
        """optimize the position of the black dots

        Takes the default position of black dots and applies simple tranformations
        to these positions. Calculate the `penalty', i.e., how well do
        transformed black dots describe the reality

        calls `new_position_of_dot`
        calls `total_penalty_for_single_dot`

        ****************************************************
        # Pass this function to minimization routine. For November 2021 run I used:
        bounds = np.array([(-0.5,0.5), (-0.5,0.5), (-0.005,0.005),
                           (-0.2,0.2), (-200,200), (-200,200)])
        # spawn the initial simples to search for the solutions
        init_simplex = np.zeros((7, 6))
        for l in range(1,7):
            init_simplex[l] = bounds[:, 0]+(bounds[:, 1]-bounds[:, 0])*np.random.random_sample(size=6)
        init_simplex[0]=[  0.23196286,  -0.11102326,  -0.00062335,
                         -0.00675098, -78.35643258,  69.39903929]

        # simple Nelder-Mead will do sufficently good job
        t1 = time.time()
        res = scipy.optimize.minimize(optimize_black_dots_instance.optimize_function,
                                      x0=init_simplex[0], method='Nelder-Mead',
                                      options={'maxiter':1000, 'initial_simplex':init_simplex})
        ****************************************************

        Parameters
        ----------
        design_variables: `np.array?`
            variables that describe the transformation
        return_full_result: `bool`
            change the amount of output

        Returns
        ----------
        if return_full_result == False:
           single float with total penalty
        if return_full_result == True:
           array with all dots, list contaning [index,
           penalty before optimization, penalty after optimization]
        """

        xd_mod = design_variables[0]
        yd_mod = design_variables[1]
        scale = design_variables[2]
        rot = design_variables[3]
        x_scale_rot = design_variables[4]
        y_scale_rot = design_variables[5]
        list_of_test_dots = []
        list_of_total_penalty_for_single_dot = []
        for i in np.arange(0+1, 2394+1):
            try:
                new_position_for_single_dot = self.new_position_of_dot(i, xd_mod, yd_mod,
                                                                       scale, rot, x_scale_rot, y_scale_rot)
                list_of_total_penalty_for_single_dot.append(
                    self.total_penalty_for_single_dot(i, new_position_for_single_dot[0],
                                                      new_position_for_single_dot[1]))
                list_of_test_dots.append(i)
            except IndexError:
                pass
        if return_full_result is False:
            return np.sum(list_of_total_penalty_for_single_dot)
        else:
            return np.array(list_of_test_dots), list_of_total_penalty_for_single_dot #noqa: W292