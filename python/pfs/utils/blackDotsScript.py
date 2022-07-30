# from pfs.utils.optimizeBlackDots import OptimizeBlackDots
from optimizeBlackDots import OptimizeBlackDots
from pfs.datamodel import PfsDesign
from pfs.utils.fiberids import FiberIds

import numpy as np
import pandas as pd
from scipy import optimize

import argparse
import ast

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# from matplotlib.legend_handler import HandlerPatch
np.set_printoptions(suppress=True)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams['figure.figsize'] = (14, 10)

gfm = pd.DataFrame(FiberIds().data)
fbi = FiberIds()

"""
Example use:
python blackDotsScript.py -work_directory /work/ncaplar/BlackDotsData/ -first_name mcsData.v077873.phi \
-second_name mcsData.v077875.theta -first_type phi -second_type theta -theta_design 4090361041684486814 \
-phi_design 5426181922868053827 -broken_design False -cobra_id 505

Arguments:
-work_directory: Location where the crossing data, dots and CobraGeometry is stored
                 Also where the new dots and auxiliary outputs will be stored
-first_name: Name of the first crossing numpy file (specify without .npy at the end)
-second_name: Name of the second crossing numpy file (specify without .npy at the end)
-first_type: Which move is the first crossing (theta, phi)
-second_type: Which move is the second crossing (theta, phi)
-theta_design: Name of the design file for theta crossing
-phi_design: Name of the design file for phi crossing
-broken_design: Is the design ``broken''. True for below 077873, False for later visits
-cobra_id: Which cobra to visualize
"""


class OptimizeBlackDotsVisualize:
    """Visualize the position of black dots

    Class with function for visualization of the positions
    of black dots for PFS spectrograph

    Parameters
    ----------
    optimize_black_dots_instance: `pfs.utils.optimizeBlackDots.OptimizeBlackDots`
        Optimization class instance
    getCobraGeometry_cobra_id: `pd.DataFrame`
        Contains informatin about positions of cobras
    """

    def __init__(self, mcs_data_all_1, mcs_data_all_2,
                 optimize_black_dots_instance, getCobraGeometry_cobra_id,
                 theta_design, phi_design, broken_design=False):

        self.dots_new = optimize_black_dots_instance.dots_new
        self.getCobraGeometry_cobra_id = getCobraGeometry_cobra_id
        self.dots = optimize_black_dots_instance.dots
        self.obs_and_predict_multi = optimize_black_dots_instance.obs_and_predict_multi
        self.mcs_data_all_1 = np.copy(mcs_data_all_1)
        self.mcs_data_all_2 = np.copy(mcs_data_all_2)
        self.theta_design = theta_design
        self.phi_design = phi_design
        self.broken_design = broken_design

        self.mcs_data_all_1 = optimize_black_dots_instance.list_of_mcs_data_all[0]
        self.mcs_data_all_2 = optimize_black_dots_instance.list_of_mcs_data_all[1]

        # optimization_result = optimize_black_dots_instance.optimization_result
        res = optimize_black_dots_instance.res
        self.res = res

        # compare the optimized results with the non-optimized result
        optimize_black_dots_instance.optimize_function([1, 0, 0, 0, 0, 0,
                                                        1, 0, 0, 0, 0, 0,
                                                        0, 0, 0, 0])
        list_of_total_penalty_for_single_dot_original = optimize_black_dots_instance.optimization_result
        optimize_black_dots_instance.optimize_function(res.x)
        list_of_total_penalty_for_single_dot_optimized = optimize_black_dots_instance.optimization_result

        list_of_total_penalty_for_single_dot_original = np.sum(list_of_total_penalty_for_single_dot_original,
                                                               axis=0)
        self.list_of_total_penalty_for_single_dot_original = list_of_total_penalty_for_single_dot_original
        list_of_total_penalty_for_single_dot_optimized =\
            np.sum(list_of_total_penalty_for_single_dot_optimized, axis=0)
        self.list_of_total_penalty_for_single_dot_optimized = list_of_total_penalty_for_single_dot_optimized

    def show_single_cobra(self, cobra_id, show_predictions=True, show_optimization=True):
        """Visualize position of a single black spot

        Parameters
        ----------
        cobra_id: `int`
            index of cobra
        show_predictions: `bool`
            Show predicted positions of cobra behind the black dot
        show_optimization: `bool`
            Show optimized position of the black dot

        Returns
        ----------
        Figure
        """

        isGood_1 = self.obs_and_predict_multi[0][2][cobra_id].astype(bool)
        isGood_2 = self.obs_and_predict_multi[1][2][cobra_id].astype(bool)

        # actual_position_x_1_observed = self.obs_and_predict_multi[0][0][cobra_id][isGood_1]
        # actual_position_y_1_observed = self.obs_and_predict_multi[0][1][cobra_id][isGood_1]
        # actual_position_x_2_observed = self.obs_and_predict_multi[1][0][cobra_id][isGood_2]
        # actual_position_y_2_observed = self.obs_and_predict_multi[1][1][cobra_id][isGood_2]

        predicted_position_x_1_not_observed = self.obs_and_predict_multi[0][0][cobra_id][~isGood_1]
        predicted_position_y_1_not_observed = self.obs_and_predict_multi[0][1][cobra_id][~isGood_1]
        predicted_position_x_2_not_observed = self.obs_and_predict_multi[1][0][cobra_id][~isGood_2]
        predicted_position_y_2_not_observed = self.obs_and_predict_multi[1][1][cobra_id][~isGood_2]

        xd_original, yd_original = self.dots.iloc[cobra_id][['x', 'y']].values
        xd_modified, yd_modified = self.dots_new.iloc[cobra_id][['x', 'y']].values
        thetaDesign = PfsDesign.read(self.theta_design, dirName='/data/pfsDesign/')
        phiDesign = PfsDesign.read(self.phi_design, dirName='/data/pfsDesign/')

        print(self.broken_design)
        print(type(self.broken_design))
        if self.broken_design:
            phiDesign = phiDesign.pfiNominal[~np.isnan(phiDesign.pfiNominal[:, 0])]
            thetaDesign = thetaDesign.pfiNominal[~np.isnan(thetaDesign.pfiNominal[:, 0])]
            phiX, phiY = phiDesign[cobra_id]
            thetaX, thetaY = thetaDesign[cobra_id]
        else:
            fiberId, = gfm.query(f'cobraId=={cobra_id+1}').fiberId
            print(fiberId)
            [[phiX, phiY]] = phiDesign.pfiNominal[phiDesign.fiberId == fiberId]
            [[thetaX, thetaY]] = thetaDesign.pfiNominal[thetaDesign.fiberId == fiberId]

        xc_1, yc_1, R_1, res_1 = find_xc_yc_r(self.mcs_data_all_1, cobra_id + 1)
        xc_2, yc_2, R_2, res_2 = find_xc_yc_r(self.mcs_data_all_2, cobra_id + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12))

        data1 = ax1.scatter(self.mcs_data_all_1[0, cobra_id+1], self.mcs_data_all_1[1, cobra_id+1],
                            color='blue', label='observed in run 1')
        ax1.scatter(self.mcs_data_all_2[0, cobra_id+1], self.mcs_data_all_2[1, cobra_id+1],
                    color='orange')
        ax1.scatter(self.getCobraGeometry_cobra_id.loc[cobra_id+1]['center_x_mm'],
                    self.getCobraGeometry_cobra_id.loc[cobra_id+1]['center_y_mm'], color='black')
        cc_cobra = Circle((self.getCobraGeometry_cobra_id.loc[cobra_id+1]['center_x_mm'],
                           self.getCobraGeometry_cobra_id.loc[cobra_id+1]['center_y_mm']),
                          self.getCobraGeometry_cobra_id.loc[cobra_id+1]['motor_theta_length_mm'],
                          color='black', alpha=0.25, label='theta radius')
        ax1.scatter(xd_original, yd_original, color='black', s=200)
        if show_optimization:
            ax1.scatter(xd_modified, yd_modified, color='black', s=200)

        cc_original = Circle((xd_original, yd_original), 0.75, color='red',
                             alpha=0.15, label='original dot position')
        cc_modified = Circle((xd_modified, yd_modified), 0.75, color='red',
                             alpha=1, label='optimized dot position', fill=False, lw=5)

        cc_blue = Circle((xc_1, yc_1), R_1, color='blue',
                         alpha=0.5, label='original dot position', fill=False, lw=2)
        center_cc_blue = ax1.scatter(xc_1, yc_1,
                                     color='blue', label='center of circle 1', marker='p', s=200)
        cc_orange = Circle((xc_2, yc_2), R_2, color='orange',
                           alpha=0.5, label='original dot position', fill=False, lw=2)
        center_cc_orange = ax1.scatter(xc_2, yc_2,
                                       color='orange', label='center of circle 2', marker='p', s=200)

        ax1.set_aspect(1)
        ax1.add_artist(cc_cobra)
        ax1.add_artist(cc_original)
        ax1.add_artist(cc_blue)
        ax1.add_artist(cc_orange)
        if show_optimization:
            ax1.add_artist(cc_modified)

        if show_predictions:
            ax1.scatter(predicted_position_x_1_not_observed, predicted_position_y_1_not_observed,
                        s=60, facecolors='none', edgecolors='blue')
            ax1.scatter(predicted_position_x_2_not_observed, predicted_position_y_2_not_observed,
                        s=60, facecolors='none', edgecolors='orange')
        if show_optimization:
            ax1.arrow(xd_original, yd_original, xd_modified-xd_original, yd_modified-yd_original,
                      head_width=0.04, length_includes_head=True)

        ax1.set_ylim(yd_original-1, yd_original+1)
        ax1.set_xlim(xd_original-1, xd_original+1)
        ax1.set_xlabel('x position [mm]')
        ax1.set_ylabel('y position [mm]')

        cc_cobra = Circle((self.getCobraGeometry_cobra_id.loc[cobra_id+1]['center_x_mm'],
                           self.getCobraGeometry_cobra_id.loc[cobra_id+1]['center_y_mm']),
                          self.getCobraGeometry_cobra_id.loc[cobra_id+1]['motor_theta_length_mm'],
                          color='black', alpha=0.25, label=r'cobra $\theta$ radius')
        cc_original = Circle((xd_original, yd_original), 0.75, color='red',
                             alpha=0.15, label='original dot position')
        cc_modified = Circle((xd_modified, yd_modified), 0.75, color='red',
                             alpha=1, label='optimized dot position', fill=False, lw=5)

        cc_blue = Circle((xc_1, yc_1), R_1, color='blue',
                         alpha=0.5, label='original dot position', fill=False, lw=2)
        center_cc_blue = ax2.scatter(xc_1, yc_1,
                                     color='blue', label='center of circle 1', marker='p', s=200)
        cc_orange = Circle((xc_2, yc_2), R_2, color='orange',
                           alpha=0.5, label='original dot position', fill=False, lw=2)
        center_cc_orange = ax2.scatter(xc_2, yc_2,
                                       color='orange', label='center of circle 2', marker='p', s=200)

        ax2.scatter(self.getCobraGeometry_cobra_id.loc[cobra_id+1]['center_x_mm'],
                    self.getCobraGeometry_cobra_id.loc[cobra_id+1]['center_y_mm'], color='black')
        ax2.scatter(xd_original, yd_original, color='black')
        if show_optimization:
            ax2.scatter(xd_modified, yd_modified, color='black')

        ax2.set_aspect(1)
        ax2.add_artist(cc_cobra)
        ax2.add_artist(cc_original)
        ax2.add_artist(cc_blue)
        ax2.add_artist(cc_orange)
        if show_optimization:
            ax2.add_artist(cc_modified)

        data1 = ax2.scatter(self.mcs_data_all_1[0, cobra_id+1], self.mcs_data_all_1[1, cobra_id+1],
                            color='blue', label='observed in run 1')
        data2 = ax2.scatter(self.mcs_data_all_2[0, cobra_id+1], self.mcs_data_all_2[1, cobra_id+1],
                            color='orange', label='observed in run 2')
        theta_start = ax2.scatter(thetaX, thetaY, label='starting for theta', color='orange', s=250)
        phi_start = ax2.scatter(phiX, phiY, label='starting for phi', color='blue', s=250)

        if show_predictions:
            predicted1 = ax2.scatter(predicted_position_x_1_not_observed, predicted_position_y_1_not_observed,
                                     s=60, facecolors='none', edgecolors='blue',
                                     label='predicted for run 1')
            predicted2 = ax2.scatter(predicted_position_x_2_not_observed, predicted_position_y_2_not_observed,
                                     s=60, facecolors='none', edgecolors='orange',
                                     label='predicted for run 2')

        if show_optimization:
            ax2.arrow(xd_original, yd_original, xd_modified-xd_original, yd_modified-yd_original,
                      head_width=0.04, length_includes_head=True)

        ax2.set_ylim(yd_original-5, yd_original+5)
        ax2.set_xlim(xd_original-5, xd_original+5)
        ax2.set_xlabel('x position [mm]')
        ax2.set_ylabel('y position [mm]')

        ax2.legend(handles=[cc_cobra, cc_original, cc_modified, data1, data2, predicted1, predicted2,
                            phi_start, theta_start, center_cc_blue, center_cc_orange])
        fiberId = fbi.fiberId[fbi.cobraId == (cobra_id+1)][0]
        fig.savefig(work_directory + 'show_single_cobra_' + str(cobra_id)+'.png', bbox_inches='tight')

    def show_improvment_1d(self):
        """Show the improvment of the quality measure

        Parameters
        ----------

        Returns
        ----------
        Figure
        """
        figure, axes = plt.subplots(figsize=(26, 6))
        plt.subplot(131)
        plt.hist(np.array(np.array(self.list_of_total_penalty_for_single_dot_original)),
                 bins=np.arange(-10, 40, 1))
        plt.title("'Penalty' for the original dot positions")
        plt.ylim(0, 1400)
        plt.subplot(132)
        plt.hist(np.array(self.list_of_total_penalty_for_single_dot_optimized),
                 bins=np.arange(-10, 40, 1))
        plt.title("'Penalty' for the optimized dot positions")
        plt.ylim(0, 1400)
        plt.subplot(133)
        plt.hist(-np.array(np.array(self.list_of_total_penalty_for_single_dot_original)
                           - np.array(self.list_of_total_penalty_for_single_dot_optimized)),
                 bins=np.arange(-20, 10, 1))
        plt.ylim(0, 1400)
        plt.title("Improvement due to optimization")
        # in the final plot, on the right hand side,
        # positive numbers are bad, negative values are good

        plt.savefig(work_directory + 'show_improvment_1d.png', bbox_inches='tight')

    def show_improvment_2d(self):
        """Show the improvment of the quality measure as it appears on the focal plane

        Parameters
        ----------

        Returns
        ----------
        Figure
        """

        xd_original_test_dots_survived = self.dots['x']
        yd_original_test_dots_survived = self.dots['y']

        # cmap = matplotlib.cm.get_cmap('jet')
        # colors_orignal = cmap(np.array(self.list_of_total_penalty_for_single_dot_original)/15)
        # colors_optimized  = cmap(np.array(self.list_of_total_penalty_for_single_dot_optimized)/15)
        # overloading the definition for cmap
        # cmap = matplotlib.cm.get_cmap('bwr')
        # colors_difference = cmap(np.array(self.list_of_total_penalty_for_single_dot_original)/15
        #                          - np.array(self.list_of_total_penalty_for_single_dot_optimized)/15 + 0.5)

        try:
            from collections.abc import Callable  # noqa
        except ImportError:
            from collections import Callable  # noqa

        matplotlib.rcParams.update({'font.size': 18})
        plt.rcParams["figure.facecolor"] = "white"
        # figure = plt.figure(figsize=(28, 8))

        plt.subplot(131)
        plt.title("'Penalty' for the original dot positions")
        sc = plt.scatter(xd_original_test_dots_survived, yd_original_test_dots_survived,
                         s=80, c=np.array(self.list_of_total_penalty_for_single_dot_original),
                         lw=2, edgecolors='black', cmap=matplotlib.cm.get_cmap('jet'), vmax=25)
        plt.colorbar(sc, fraction=0.046, pad=0.04)
        plt.gca().set_aspect('equal')
        plt.ylabel('y position [mm]')
        plt.xlabel('x position [mm]')

        plt.subplot(132)
        plt.title("'Penalty' for the optimized dot positions")
        sc2 = plt.scatter(xd_original_test_dots_survived, yd_original_test_dots_survived,
                          s=80, c=np.array(self.list_of_total_penalty_for_single_dot_optimized),
                          lw=2, edgecolors='black', cmap=matplotlib.cm.get_cmap('jet'), vmax=25)
        plt.gca().set_aspect('equal')
        plt.colorbar(sc2, fraction=0.046, pad=0.04)
        plt.ylabel('y position [mm]')
        plt.xlabel('x position [mm]')

        plt.subplot(133)
        plt.title("Improvement due to optimization")
        sc3 = plt.scatter(xd_original_test_dots_survived, yd_original_test_dots_survived,
                          s=80, c=np.array(self.list_of_total_penalty_for_single_dot_original)/1
                          - np.array(self.list_of_total_penalty_for_single_dot_optimized)/1, lw=2,
                          edgecolors='black', cmap=matplotlib.cm.get_cmap('bwr'), vmax=25, vmin=-25)
        plt.gca().set_aspect('equal')
        plt.colorbar(sc3, fraction=0.046, pad=0.04)
        plt.ylabel('y position [mm]')
        plt.xlabel('x position [mm]')

        plt.savefig(work_directory + 'show_improvment_2d.png', bbox_inches='tight')

    def show_movement(self):
        """Show the change of the inferred positions of the black spots

        Parameters
        ----------

        Returns
        ----------
        Figure
        """
        list_of_change = []

        xd_original = self.dots['x']
        yd_original = self.dots['y']

        xd_new = self.dots_new['x']
        yd_new = self.dots_new['y']

        list_of_change.append([np.array((xd_original, yd_original)),
                               np.array((xd_new, yd_new)) - np.array((xd_original, yd_original))])

        array_of_change = np.array(list_of_change)

        plt.rcParams["figure.facecolor"] = "white"
        plt.figure(figsize=(12, 12))
        plt.quiver(array_of_change[:, 0][:, 0], array_of_change[:, 0][:, 1],
                   +array_of_change[:, 1][:, 0], array_of_change[:, 1][:, 1],
                   units='width', angles='xy', scale_units='xy', scale=0.02, width=0.002, headwidth=5.5)

        plt.quiver(140, 210, 0.5, 0,
                   units='width', angles='xy', scale_units='xy', scale=0.02, width=0.002, headwidth=5.5)
        plt.title("Movement of center of the dot in the optimization")
        plt.text(140, 215, s='0.5 mm')
        plt.axvline(0, color='brown', ls='--')
        plt.axhline(0, color='brown', ls='--')
        plt.gca().set_aspect('equal')

        plt.xlim(-250, 250)
        plt.ylim(-250, 250)

        plt.savefig(work_directory + 'show_movement.png', bbox_inches='tight')


def calc_R(x, y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)


def f_2(c, x, y):
    """calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc)"""
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()


def find_xc_yc_r(mcs_data_all, cob_id):
    x_data = mcs_data_all[0][cob_id]
    y_data = mcs_data_all[1][cob_id]

    x = x_data[~np.isnan(x_data)]
    y = y_data[~np.isnan(y_data)]

    x_m = np.mean(x)
    y_m = np.mean(y)

    center_estimate = x_m, y_m
    center_2, ier = optimize.leastsq(f_2, center_estimate, args=(x, y))

    xc_2, yc_2 = center_2
    Ri_2 = calc_R(x, y, *center_2)
    R_2 = Ri_2.mean()
    residu_2 = sum((Ri_2 - R_2)**2)

    return xc_2, yc_2, R_2, residu_2


parser = argparse.ArgumentParser(description="Starting args import",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog='Done with import')

parser.add_argument(
    "-work_directory",
    help="where is the data that you wish to analyze",
    type=str)

parser.add_argument(
    "-first_name",
    help="name of the first crossing file",
    type=str)

parser.add_argument(
    "-second_name",
    help="name of the second crossing file",
    type=str)

parser.add_argument(
    "-first_type",
    help="type of the first crossing file",
    type=str)

parser.add_argument(
    "-second_type",
    help="type of the second crossing file",
    type=str)

parser.add_argument(
    "-theta_design",
    help="theta design integer",
    type=int)

parser.add_argument(
    "-phi_design",
    help="theta design integer",
    type=int)

parser.add_argument(
    "-broken_design",
    help="True before 077873",
    type=str)

parser.add_argument(
    "-cobra_id",
    help="which cobra to visualize",
    type=int)


# Finished with specifying arguments

################################################
# Assigning arguments to variables
args = parser.parse_args()


work_directory = args.work_directory.strip()
first_name = args.first_name.strip()
second_name = args.second_name.strip()
first_type = args.first_type.strip()
second_type = args.second_type.strip()
theta_design = args.theta_design
phi_design = args.phi_design
broken_design = ast.literal_eval(args.broken_design)
cobra_id = args.cobra_id

# positions of the dots
dots = pd.read_pickle(work_directory + "dots.pkl")

# original position of the cobras
# only needed for plotting
getCobraGeometry_cobra_id = pd.read_pickle(work_directory + "getCobraGeometry_cobra_id.pkl")

mcs_data_all_1 = np.load(work_directory + first_name + '.npy')
if first_name == 'mcs_data_all_1':
    # take only 40 measurments - 41st was wrongly attached to this file
    # this is the emergency fix to analyze the dataset from November 2021
    mcs_data_all_1 = mcs_data_all_1[:, :, :40]

mcs_data_all_2 = np.load(work_directory + second_name + '.npy')


# Running the main script
list_of_mcs_data_all = [mcs_data_all_1, mcs_data_all_2]
list_of_descriptions = [first_type, second_type]
optimize_black_dots_instance = OptimizeBlackDots(dots, list_of_mcs_data_all,
                                                 list_of_descriptions,
                                                 outlier_distance=4.5,
                                                 max_break_distance=1.7,
                                                 residual_max=0.3,
                                                 min_n_points=20,
                                                 order_phi_inter=4,
                                                 order_phi_extra=2)
dots_new = optimize_black_dots_instance.find_optimized_dots(scal_var=np.array([1, 0, 0, 0, 0, 0,
                                                                               1, 0, 0, 0, 0, 0,
                                                                               0, 0, 0, 0]),
                                                            max_iter=10000, rand_val=1234)

optimization_result = optimize_black_dots_instance.optimization_result
res = optimize_black_dots_instance.res

# save the new position of the dots
dots_new.to_pickle(work_directory+'dots_new.pkl')

penalty_before = optimize_black_dots_instance.optimize_function([1, 0, 0, 0, 0, 0,
                                                                 1, 0, 0, 0, 0, 0,
                                                                 0, 0, 0, 0])
print("penalty before the optimization: "+str(penalty_before))
penalty_after = optimize_black_dots_instance.optimize_function(res.x)
print("penalty after the optimization: "+str(penalty_after))

# select  instance of optimization to visualize
OptimizeBlackDotsVisualize_instance = OptimizeBlackDotsVisualize(mcs_data_all_1, mcs_data_all_2,
                                                                 optimize_black_dots_instance,
                                                                 getCobraGeometry_cobra_id,
                                                                 theta_design=theta_design,
                                                                 phi_design=phi_design,
                                                                 broken_design=broken_design)

single_cobra_plot = OptimizeBlackDotsVisualize_instance.show_single_cobra(cobra_id=int(cobra_id))
show_improvment_1d = OptimizeBlackDotsVisualize_instance.show_improvment_1d()
show_improvment_2d = OptimizeBlackDotsVisualize_instance.show_improvment_2d()
show_movement = OptimizeBlackDotsVisualize_instance.show_movement()
