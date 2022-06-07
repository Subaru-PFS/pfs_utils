#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
from lsst.geom import AffineTransform
from pfs.drp.stella.math import evaluateAffineTransform


class OptimizeBlackDots:
    """Optimization of position of black dots

    Class with functions for modifying the positions of black dots
    for PFS spectrograph

    Parameters
    ----------
    dots : `pandas.dataframe`, (N_fib, 4)
        Dataframe contaning initial guesses for dots
        The columns are ['spotId', 'x', 'y', 'r']
    list_of_mcs_data_all : `list` of `np.arrays`
        List contaning arrays with x and y positions
        of moving cobra from mcs observations
        Each array with a shape (2, N_fib + 1, N_obs)
    list_of_descriptions : `list` of `str`
        What kind of moves have been made (theta or phi)

    Examples
    ----------
    # example for November 2021 run to create
    # list_of_mcs_data_all and list_of_descriptions
    # which are inputs for the class

    >>> list_of_mcs_data_all = [mcs_data_all_1, mcs_data_all_2]
    >>> list_of_descriptions = ['theta', 'phi']
    """

    def __init__(self, dots, list_of_mcs_data_all, list_of_descriptions, **kwargs):
        self.dots = dots
        self.n_runs = len(list_of_descriptions)
        self.number_of_fibers = 2394
        # radius of black dots is 0.75 mm
        # small variations exist, but those have neglible impact
        self.radius_of_black_dots = 0.75

        self.list_of_mcs_data_all = list_of_mcs_data_all
        obs_and_predict_multi = []
        for obs in range(len(list_of_mcs_data_all)):
            mcs_data_all = list_of_mcs_data_all[obs]
            description = list_of_descriptions[obs]
            obs_and_predict_single =\
                self.predict_positions(mcs_data_all, description, **kwargs)
            obs_and_predict_multi.append(obs_and_predict_single)
        self.obs_and_predict_multi = obs_and_predict_multi

    def predict_positions(self, mcs_data_all, type_of_run='theta',
                          outlier_distance=2, max_break_distance=1.7,
                          min_n_points=10, residual_max=0.5,
                          order_theta_inter=4, order_theta_extra=2,
                          order_phi_inter=2, order_phi_extra=1, **kwargs):
        """Predict positions for spots which were not observed

        Parameters
        ----------
        mcs_data_all: `np.array`, (2, N_fib + 1, N_obs)
           Positions of observed spots
        type_of_run: `str`
            Describe which kind of movement is being done
            {'theta', 'phi'}
        outlier_distance: `float`, optional
            Points more than outlier_distance from
            the median of all points will be replaced by nan [mm]
        max_break_distance: `float`, optional
            Maximal distance between points on the
            either side of the black dot [mm]
        min_n_points: `int`, optional
            Minimal number of points seen in a single crossing in order
            to predict the trajectory for non-seen points
        residual_max: `float`, optional
            Max residual to the poly fit to consider the fit sucessful
        order_theta_inter: `int`, optional
            Order of poly fit for interpolating predictions for theta run
        order_theta_extra: `int`, optional
            Order of poly fit for extrapolating predictions for theta run
        order_phi_inter: `int`, optional
            Order of poly fit for interpolating predictions for phi run
        order_phi_extra: `int`, optional
            Order of poly fit for extrapolating predictions for phi run

        Returns
        ----------
        mcs_data_extended `np.array`, (3, N_fiber, N_obs)
            Positions of observed and predicted spots.
            Has an additional dimension as compared to mcs_data_all,
            to indicate if the obervations was `good`
            (seen in the data, indicated with 1.)
            or not (not seen in the data, indicated with 0.)

        Notes
        ----------
        Creates predictions for where the points should be,
        when you do not see points.

        The data is first cleaned. Points more than outlier_distance
        are removed as they have been atributed to wrong cobras.

        If the predicted movement is too unreasonable, we disregard
        that prediction. The diameter of a dot is 1.5 mm,
        so the distance can not be larger than that,
        or we would see the data on the other side of the dot.
        I have placed 1.7 mm as a limit
        to account for possible variations of size and measurment errors.

        After cleaning of the data, the algoritm searches to see
        if there are breaks in the observed data.

        Depending on if you see points from both sides of the black dots,
        and depending on the type of movement, the algorithm changes the
        order of complexity for the extrapolation/interpolation. Theta
        moves are better behaved so more flexibility is fitting is possible.
        It is also easier to interpolate than extrapolate, so we allow for
        more freedom.

        Improvents to the cleaning algorithm are definetly possible. Few spots
        have observations which are obviously coming from different fibers.
        Some fibers have breaks from either side of observation for unknown
        reasons, and this could be accounted as well.
        """
        valid_run = {'theta', 'phi'}
        if type_of_run not in valid_run:
            raise ValueError("results: status must be one of %r." % valid_run)

        n_obs = mcs_data_all.shape[2]
        mcs_data_extended = np.full((3, self.number_of_fibers, n_obs), np.nan)

        for i in range(self.number_of_fibers):
            fib = i + 1
            x_measurments = mcs_data_all[0, fib]
            y_measurments = mcs_data_all[1, fib]

            # index of the points which are observed
            isGood_init = np.isfinite(x_measurments) & np.isfinite(y_measurments)

            if np.sum(isGood_init) < min_n_points:
                mcs_data_extended[0, i][isGood_init] = mcs_data_all[0, fib][isGood_init]
                mcs_data_extended[1, i][isGood_init] = mcs_data_all[1, fib][isGood_init]
                predicted_position_x = np.full(n_obs, np.nan)
                predicted_position_y = np.full(n_obs, np.nan)
                mcs_data_extended[0, i][~isGood_init] = predicted_position_x[~isGood_init]
                mcs_data_extended[1, i][~isGood_init] = predicted_position_y[~isGood_init]
                mcs_data_extended[2, i] = isGood_init
                continue

            # Select the point that are more than outlier_distance (default is 2 mm)
            # away from median of all of the points and replace them with np.nan
            # We assume these points have been attributed to wrong cobras
            xMedian = np.median(x_measurments[isGood_init])
            yMedian = np.median(y_measurments[isGood_init])
            large_outliers_selection =\
                (np.abs(y_measurments[isGood_init] - yMedian) > outlier_distance) |\
                (np.abs(x_measurments[isGood_init] - xMedian) > outlier_distance)

            # Remove the large outliers from the selection of good points
            isGood = np.copy(isGood_init)
            # isGood[isGood_init == 1] = isGood_init[isGood_init == 1]*~large_outliers_selection
            isGood[isGood_init == 1] &= ~large_outliers_selection

            # Replace all unsuccessful measurments wih nan
            x_measurments[~isGood] = np.nan
            y_measurments[~isGood] = np.nan
            # Specify the number of breaks in the data
            if np.sum(np.diff(isGood)) == 1:
                number_of_breaks = 1
            else:
                number_of_breaks = 2

            if type_of_run == 'theta':
                poly_order_inter = order_theta_inter
                poly_order_extra = order_theta_extra
            if type_of_run == 'phi':
                poly_order_inter = order_phi_inter
                poly_order_extra = order_phi_extra
            # Create prediction for where the points should be, when you do not see points
            # If you do not see points from both sides of the black dots, do simpler extrapolation
            # If you see points from both sides of the black dots, do more complex interpolation
            if number_of_breaks == 1:
                poly_order = poly_order_extra
            else:
                poly_order = poly_order_inter

            deg2_fit_to_single_point_y, residuals_y =\
                self.poly_fit(n_obs, y_measurments, isGood, poly_order)
            deg2_fit_to_single_point_x, residuals_x =\
                self.poly_fit(n_obs, x_measurments, isGood, poly_order)

            poly_deg_x = np.poly1d(deg2_fit_to_single_point_x)
            poly_deg_y = np.poly1d(deg2_fit_to_single_point_y)

            # Cleaning, to exclude the results where the interpolation results make no sense
            # If cobra has more than 5 observations hidden, calculate how much distance
            # we predicted that the cobra has moved during the `predicted` movement,
            # i.e., while it was not observed. This will be used
            # below to remove prediction that make no sense
            x_predicted = poly_deg_x(np.arange(0, n_obs))[~isGood]
            y_predicted = poly_deg_y(np.arange(0, n_obs))[~isGood]
            if np.sum(~isGood) > 5:
                r_distance_predicted = np.hypot(x_predicted[-1] - x_predicted[0],
                                                y_predicted[-1] - y_predicted[0])
            else:
                r_distance_predicted = 0

            # Residuals from the polyfit are larger than 0.5 mm only in cases of
            # catastrophic failure
            if (residuals_y[0] < residual_max and residuals_x[0] < residual_max
                    and ~((number_of_breaks == 1) and (r_distance_predicted > max_break_distance))):
                predicted_position_x = poly_deg_x(np.arange(0, n_obs))
                predicted_position_y = poly_deg_y(np.arange(0, n_obs))
            else:
                predicted_position_x = np.full(n_obs, np.nan)
                predicted_position_y = np.full(n_obs, np.nan)

            mcs_data_extended[0, i][isGood] = mcs_data_all[0, fib][isGood]
            mcs_data_extended[1, i][isGood] = mcs_data_all[1, fib][isGood]
            mcs_data_extended[0, i][~isGood] = predicted_position_x[~isGood]
            mcs_data_extended[1, i][~isGood] = predicted_position_y[~isGood]
            mcs_data_extended[2, i] = isGood

        return mcs_data_extended

    @staticmethod
    def poly_fit(n_obs, measurments, isGood, poly_order):
        """Fit polynomial fit to the measurments

        Parameters
        ----------
        n_obs: `int`
            Number of observations
        measurments: `np.array`
            Measurments of the crossing experiment
        isGood: `np.array`
            Array contaning information if the fiber was seen
            in this particular observation
        poly_order: `int`
            Which poly order to fit to the measured data

        Returns
        ----------
        fit_to_single_point: `np.array`
            Polynomials factors of the fit
        residuals: `np.array`
            Residual to the fit

        Notes
        ----------
        Gets called by `predict_positions`

        """
        fit_for_single_fiber, residuals =\
            np.polyfit(np.arange(0, n_obs)[isGood],
                       measurments[isGood], poly_order, full=True)[0:2]

        return fit_for_single_fiber, residuals

    @staticmethod
    def check(x_obs, y_obs, xd, yd, r):
        """Check if the cobra is covered by the black dot

        Parameters
        ----------
        x_obs: `np.array` of `float`, (N_fib, N_obs)
            x coordinates of the observations [mm]
        y_obs: `np.array` of `float``, (N_fib, N_obs)
            y coordinate of the observations [mm]
        xd: `np.array` of `float``, (N_fib,)
            x coordinate of the dot [mm]
        yd: `np.array` of `float`, (N_fib,)
            y coordinate of the dot [mm]
        r: `float`
            Radius of dot [mm]

        Returns
        ----------
        cover value: `bool`
            Return True if the cobra is covered

        Notes
        ----------
        Gets called by `total_penalty_for_single_dot`
        """
        x1 = (x_obs.T - xd).T
        y1 = (y_obs.T - yd).T

        return np.hypot(x1, y1) < r

    def new_position_of_dots(self, aff_mat_11, aff_mat_21, aff_mat_12,
                             aff_mat_22, aff_mat_13, aff_mat_23):
        """Move the dots to the new positions via affine tranformation

        Parameters
        ----------
        aff_mat_11: `float`
            Value for affine matrix at position [1,1]
        aff_mat_12: `float`
            Value for affine matrix at position [2,1]
        aff_mat_21: `float`
            Value for affine matrix at position [1,2]
        aff_mat_12: `float`
            Value for affine matrix at position [2,2]
        aff_mat_13: `float`
            Value for affine matrix at position [1,3]
        aff_mat_23
            Value for affine matrix at position [2,3]

        Returns
        ----------
        dots_new: `pd.dataframe`
            Dataframe contaning final guesses for dots
            The columns are ['spotId', 'x', 'y', 'r']
        """
        dots = self.dots
        xd_original = dots['x'].values
        yd_original = dots['y'].values
        transform = AffineTransform()
        AffineTransform.setParameterVector(transform,
                                           np.array([aff_mat_11, aff_mat_21,
                                                     aff_mat_12, aff_mat_22,
                                                     aff_mat_13, aff_mat_23]))
        xd_new, yd_new = evaluateAffineTransform(transform, xd_original, yd_original)
        dots_new = dots.copy(deep=True)
        dots_new['x'] = xd_new
        dots_new['y'] = yd_new

        return dots_new

    def total_penalty_for_single_dot(self, x_obs, y_obs, status, xd, yd):
        """Calculate a penalty for a single dot

        Parameters
        ----------
        x_obs: `np.array` of `float`, (N_fib, N_obs)
            x coordinates of the observations [mm]
        y_obs: `np.array` of `float`, (N_fib, N_obs)
            y coordinate of the observations [mm]
        status: `np.array` of `float`, (N_fib, N_obs)
            Status of observations
        xd: `np.array` of `float``, (N_fib,)
            x position of the dot [mm]
        yd: `np.array` of `float``, (N_fib,)
            y position of the dot [mm]

        Returns
        ----------
        total_penalty_for_single_modification: `float`
            Total penalty for a single dot, for
            dot at (xd, yd)

        Notes
        ----------
        Compare the results of coverage of input data with the moved
        dot to the actuall success status of input observations. Any discrepancy
        adds to the penalty.

        Calls `check`
        Gets called by `optimize_function`
        """
        result_check = self.check(x_obs, y_obs, xd, yd, self.radius_of_black_dots)
        total_penalty_for_single_modification = np.sum(np.array(result_check).astype(float) == status,
                                                       axis=1)

        return total_penalty_for_single_modification

    def optimize_function(self, scaling_variables):
        """Find penalty for all of fibers given the suggested moves.

        Parameters
        ----------
        design_variables: `np.array` of `floats`
            Variables that describe the transformation

        Returns
        ----------
        opt_res_summed: `float`
           Total penalty for all fibers, in all observations

        Notes
        ----------
        Takes the default position of black dots and
        applies simple tranformations to these positions.
        Calculates the `penalty', i.e., how well do
        transformed black dots describe the reality.

        This function output needs to be optimized in order to
        get best fit ot the data.

        Calls `new_position_of_dot`
        Calls `total_penalty_for_single_dot`
        Gets called by `find_optimized_dots`
        """
        xd_mod = scaling_variables[0]
        yd_mod = scaling_variables[1]
        scale = scaling_variables[2]
        rot = scaling_variables[3]
        x_scale_rot = scaling_variables[4]
        y_scale_rot = scaling_variables[5]

        optimization_result = np.zeros((self.n_runs, self.number_of_fibers))
        dots_new = self.new_position_of_dots(xd_mod, yd_mod, scale,
                                             rot, x_scale_rot, y_scale_rot)

        for run in range(self.n_runs):
            x_positions = self.obs_and_predict_multi[run][0]
            y_positions = self.obs_and_predict_multi[run][1]
            status = self.obs_and_predict_multi[run][2]

            xd = dots_new['x'].values
            yd = dots_new['y'].values
            optimization_result[run] = self.total_penalty_for_single_dot(x_positions,
                                                                         y_positions,
                                                                         status,
                                                                         xd,
                                                                         yd)

            optimization_result[run] -= np.sum(np.isnan(x_positions), 1)

        self.optimization_result = optimization_result
        return np.sum(optimization_result)

    def find_optimized_dots(self, scal_var=np.array([1, 0, 0, 1, 0, 0]),
                            scal_var_bounds=np.array([(0.99, 1.01), (-0.01, 0.01),
                                                      (-0.01, 0.01), (0.99, 1.01),
                                                      (-0.25, 0.25), (-0.25, 0.25)]),
                            max_iter=1000,
                            rand_val=1234):
        """Find the actual positions of dots

        Use Nelder-Mead algorithm to find positions of dots which minimize
        the `penalty`.

        Parameters
        ----------
        scal_var: `np.array` of `floats`, optional
            Initial guess for the parameters of the transformation
        return_full_result: `np.array`, optional
            Bounds supplied to the minimizer for the parametes
        max_iter: `int`, optional
            Maximal number of iterations for the minimizer
        rand_val: `int`, optional
            Random seed value

        Returns
        ----------
        dots_new: `pandas.dataframe`, (N_fiber, 4)
            Dataframe contaning optimized guesses for dots
            The columns are ['spotId', 'x', 'y', 'r']

        Notes
        ----------
        Calls `optimize_function`
        Calls `new_position_of_dots`
         """
        # spawn the initial simplex to search for the solutions
        init_simplex = np.zeros((7, 6))
        np.random.seed(rand_val)
        for point in range(1, 7):
            init_simplex[point] = scal_var_bounds[:, 0] +\
                (scal_var_bounds[:, 1]-scal_var_bounds[:, 0])*np.random.random_sample(size=6)
        init_simplex[0] = scal_var

        # simple Nelder-Mead will do sufficently good job
        res = minimize(self.optimize_function,
                       x0=init_simplex[0], method='Nelder-Mead',
                       options={'maxiter': max_iter, 'initial_simplex': init_simplex})

        dots_new = self.new_position_of_dots(*res.x)
        self.res = res
        return dots_new


def rotate(point, origin, angle):
    """Rotate the point around the origin

    Parameters
    ----------
    point: `tuple`
        x, y coordinate of the point to rotate
    origin: `tuple`
        x, y coordinate of the point around which to rotate
    angle: `float`
        Rotation angle [degrees]

    Returns
    ----------
    x, y: `float`, `float`
        x, y coordinate of the rotated point
    """
    radians = np.deg2rad(angle)
    x, y = point
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return qx, qy
