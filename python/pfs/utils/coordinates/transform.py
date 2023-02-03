import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pfs.utils.display import CircleHandler

__all__ = ["matchIds", "makePfiTransform", "MeasureDistortion"]

from .CoordTransp import CoordinateTransform


def matchIds(u, v, x, y, fid, matchRadius=2):
    """Match set of measured points (u, v) to a set of points (x, y) with ids fid

    If there are multiple possible matches from (u, v) to (x, y), only return the best match

    Parameters
    ----------
    u : `array` of `float`
        x-positions of points to be matched
    v : `array` of `float`
        y-positions of points to be matched
    x : `array` of `float`
        known x-positions of points
    y : `array` of `float`
        known y-positions of points
    fid : `array` of `int`
        ids for points (x, y)
    matchRadius: `float`
        maximum allowed match radius

    Returns
    -------
    fids: `array` of `int` of length len(u)
        matched values of fid or -1
    distance:  array of `float`  of length len(u)
        distance from (u, v) to nearest element of (x, y)
    """
    fid_out = np.full_like(u, -1, dtype=fid.dtype)
    distance = np.empty_like(u)

    matched_distances = {}

    for i, (up, vp) in enumerate(zip(u, v)):
        d = np.hypot(x - up, y - vp)
        distance[i] = min(d)
        if distance[i] < matchRadius:
            matched_fid = fid[np.argmin(d)]

            if matched_fid in matched_distances:  # we've already found a match
                if distance[i] > matched_distances[matched_fid]: # the old one is better
                    continue

                fid_out[fid_out == matched_fid] = -1 # invalidate old match

            matched_distances[matched_fid] = distance[i]
            fid_out[i] = matched_fid

    return fid_out, distance


def makePfiTransform(cameraName, *args, **kwargs):
    """ Transform factory, construct pfiTransform from the camera name.

    Parameters
    ----------
    cameraName : `str`
        camera used to measure fiducials position.
    """
    if cameraName is None or cameraName.lower() == "simple":
        return SimpleTransform(*args, **kwargs)
    elif 'canon' in cameraName.lower():
        return PfiTransform(*args, **kwargs)
    elif 'rmod' in cameraName.lower():
        return ASRD71MTransform(*args, **kwargs)
    elif 'usmcs' in cameraName.lower():
        return USMCSTransform(*args, **kwargs)
    else:
        raise ValueError(f'unknown transform for camera : {cameraName}')

fromCameraName = makePfiTransform       # old name, meaningless if imported from this module


class MeasureDistortion:
    def __init__(self, x, y, fid, x_mm, y_mm, fiducialId, nsigma=None, alphaRot=0.0):
        """x, y: measured positions in pfi coordinates
        fid: fiducial ids for (x, y) (-1: unidentified)
        x_mm, y_mm : true positions in pfi coordinates
        fiducialId: fiducial ids for (x_mm, y_mm)
        nsigma: clip the fitting at this many standard deviations (None => 5).  No clipping if <= 0
        alphaRot: coefficient for the dtheta^2 term in the penalty function
        """
        self.nsigma = 5 if nsigma is None else nsigma
        self.alphaRot = alphaRot

        _x = []
        _y = []
        _x_mm = []
        _y_mm = []
        #
        # The correct number of initial values; must match code in __call__()
        #
        x0, y0, theta, dscale, scale2 = np.array([0, 0, 0, 0, 0], dtype=float)
        self._args = np.array([x0, y0, theta, dscale, scale2])
        self.frozen = np.zeros(len(self._args), dtype=bool)

        for fi in fiducialId:
            ix = np.where(fi == fid)[0]
            if len(ix) > 0:
                _x.append(x[ix][0])
                _y.append(y[ix][0])
                _x_mm.append(x_mm[fiducialId == fi][0])
                _y_mm.append(y_mm[fiducialId == fi][0])

        self.x = np.array(_x)
        self.y = np.array(_y)

        self.x_mm = np.array(_x_mm)
        self.y_mm = np.array(_y_mm)

    @staticmethod
    def clip(d, nsigma):
        if len(d) == 0:
            return np.ones_like(d, dtype=bool)

        q25, q50, q75 = np.percentile(d, [25, 50, 75])
        std = 0.741*(q75 - q25)

        return np.abs(d - q50) < nsigma*std

    def __call__(self, args):
        tx, ty = self.distort(self.x, self.y, *args)

        d = np.hypot(tx - self.x_mm, ty - self.y_mm)

        if self.nsigma > 0:
            d = d[self.clip(d, self.nsigma)]

        penalty = np.sum(d**2)
        penalty += self.alphaRot*(args[2] - 0.0)**2 # include a prior on the rotation, args[2]

        return penalty

    def getArgs(self):
        return self._args

    def setArgs(self, *args):
        not_frozen = np.logical_not(np.array(self.frozen))
        self._args[not_frozen] = np.array(args[0])[not_frozen]

    def distort(self, x, y, *args, **kwargs):
        if args:
            args = np.array(args)
        else:
            args = self._args

        args[self.frozen] = self._args[np.array(self.frozen)]

        x0, y0, theta, dscale, scale2 = args  # must match length of self._args in __init__
        inverse = kwargs.get("inverse", False)

        theta = np.deg2rad(theta)
        c, s = np.cos(theta), np.sin(theta)

        r = np.hypot(x - x0, y - y0) if inverse else np.hypot(x, y)
        scale = (1 + dscale) + scale2*r**2

        if inverse:
            x = x - x0   # don't modify x
            y = y - y0
            s = -s    # change sign of theta
            tx = ( c*x + s*y)/scale
            ty = (-s*x + c*y)/scale
        else:
            tx = x0 + scale*( c*x + s*y)
            ty = y0 + scale*(-s*x + c*y)

        return tx, ty


class PfiTransform:
    def __init__(self, altitude=90, insrot=0, applyDistortion=True, nsigma=None, alphaRot=0):
        self.setParams(altitude, insrot, nsigma, alphaRot)

        self.applyDistortion = applyDistortion

    def setParams(self, altitude=90, insrot=0, nsigma=None, alphaRot=0):
        self.altitude = altitude
        self.insrot = insrot
        self.nsigma = nsigma
        self.alphaRot = alphaRot
        #
        # Unweighted linear fit to values from Jennifer:
        #
        # altitude == 90
        #   self.mcs_boresight_x_pix = 4468.6
        #   self.mcs_boresight_y_pix = 2869.6
        # altitude == 60
        #  self.mcs_boresight_x_pix = 4466.9
        #  self.mcs_boresight_y_pix = 2867.8
        # altitude == 30
        #  self.mcs_boresight_x_pix = 4474.9
        #  self.mcs_boresight_y_pix = 2849.9
        #
        self.mcs_boresight_x_pix = 4476.4 - 0.105*altitude
        self.mcs_boresight_y_pix = 2842.7 + 0.328*altitude
        #
        # Initial camera distortion; updated using updateTransform
        #
        self.mcsDistort = MeasureDistortion([], [], [], [], [], [], nsigma=0, alphaRot=self.alphaRot)

        a, b, phi = [1.882, 1.000], 0.0703, np.full(2, 32.2) + [90, 0]
        x0, y0 = a + b*np.sin(np.deg2rad(insrot - phi))
        theta = 0.708
        dscale = 0.00266
        scale2 = 2.38e-09
        args = [x0, y0, theta, dscale, scale2]

        self.mcsDistort.setArgs(args)

    def updateTransform(self, mcs_data, fiducials, matchRadius=2, nMatchMin=0.75, fig=None):
        """Update our estimate of the transform, based on the positions of fiducial fibres

        mcs_data:              Pandas DataFrame containing mcs_center_x_pix, mcs_center_y_pix
                               (measured positions in pixels in metrology camera)
        fiducials:             Pandas DataFrame containing x_mm, y_mm, fiducialId
                               (Fiducial positions in mm on PFI)
                               As returned by `butler.get("fiducials")
        matchRadius:           Radius to match points and fiducials (mm)
        nMatchMin:             Minimum number of permissible matches
                               (if <= 1, interpreted as the fraction of the number of fiducials)
        fig                    matplotlib figure for displays; or None

        Returns:
           fids:      array of `int`
                array of length mcs_data giving indices into fiducials or -1
           distance:  array of `float`
                array of length mcs_data giving distance in mm to nearest fiducial
        """
        if nMatchMin <= 1:
            nMatchMin *= len(fiducials.fiducialId)

        mcs_x_pix = mcs_data.mcs_center_x_pix.to_numpy()
        mcs_y_pix = mcs_data.mcs_center_y_pix.to_numpy()

        x_fid_mm = fiducials.x_mm.to_numpy()
        y_fid_mm = fiducials.y_mm.to_numpy()
        fiducialId = fiducials.fiducialId.to_numpy()

        # Get our best estimate of the transformed positions to give ourselves the
        # best chance of matching to the fiducial fibres
        #
        # N.b. this allows us to call updateTransform with different sets of
        # fiducials and/or configs to refine our transformation
        xd, yd = self.mcsToPfi(mcs_x_pix, mcs_y_pix)

        fid, dmin = matchIds(xd, yd, x_fid_mm, y_fid_mm, fiducialId, matchRadius=matchRadius)
        nMatch = sum(fid > 0)

        self._plotMatches(fig, x_fid_mm, y_fid_mm, xd, yd, fid, matchRadius, nMatch)

        if nMatch < nMatchMin:
            raise RuntimeError(f"I only matched {nMatch} out of {len(fiducialId)} fiducial fibres")

        applyDistortion = self.applyDistortion
        try:
            self.applyDistortion = False
            x, y = self.mcsToPfi(mcs_x_pix, mcs_y_pix)
        finally:
            self.applyDistortion = applyDistortion

        distortion = MeasureDistortion(x, y, fid, x_fid_mm, y_fid_mm, fiducialId,
                                       self.nsigma, self.alphaRot)
        distortion.frozen = self.mcsDistort.frozen

        res = scipy.optimize.minimize(distortion, distortion.getArgs(), method='Powell')
        self.mcsDistort.setArgs(res.x)

        xd, yd = self.mcsToPfi(mcs_x_pix, mcs_y_pix)
        fid, dmin = matchIds(xd, yd, x_fid_mm, y_fid_mm, fiducialId, matchRadius=matchRadius)

        self._plotMatches(fig, x_fid_mm, y_fid_mm, xd, yd, fid, matchRadius, nMatch)

        return (fid, dmin)

    #
    # Note that these two routines use instance values of mcs_boresight_[xy]_pix
    # and assume that the pfi origin is at (0, 0)
    #
    def mcsToPfi(self, x, y):
        """transform mcs pixels to pfi mm
        x, y:  position in mcs pixels

        returns:
            x, y in pfi mm
        """
        if isinstance(x, pd.Series):
            x = x.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        xyin = np.stack((x, y)).copy()

        xyin[0] = -(xyin[0] - self.mcs_boresight_x_pix)   # rotate 180 about centre
        xyin[1] = -(xyin[1] - self.mcs_boresight_y_pix)

        xy = CoordinateTransform(xyin, "mcs_pfi", za=90.0 - self.altitude, inr=self.insrot)
        xp, yp = xy[0], xy[1]

        if self.applyDistortion:
            xp, yp = self.mcsDistort.distort(xp, yp, inverse=False)

        return xp, yp

    def pfiToMcs(self, x, y, niter=5, lam=1.0):
        """transform pfi mm to mcs pixels
        x, y:  position in pfi mm
        niter: number of iterations
        lam:   convergence factor for iteration

        returns:
            x, y in mcs pixels

        The pfi_mcs transformation simply applies the scale change, whereas the mcs_pfi
        transformation includes the distortion, so we'll iterate until the computed mcs positions
        transform back to the input pfi positions
        """
        if isinstance(x, pd.Series):
            x = x.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        if self.applyDistortion:
            xm, ym = self.mcsDistort.distort(x, y, inverse=True)
        else:
            xm, ym = x.copy(), y.copy()

        xyin = np.stack((xm, ym))
        xyout = CoordinateTransform(xyin, "pfi_mcs", za=90.0 - self.altitude,
                                    inr=self.insrot) # first guess at MCS coordinates
        #
        # Deal with coordinate transformations on the MCS
        #
        xyout = -xyout  # rotate 180 about centre
        xyout[0] += self.mcs_boresight_x_pix   # include centre of mcs
        xyout[1] += self.mcs_boresight_y_pix
        #
        # This is our first guess at the MCS position, but it's not very good, so we'll iterate
        # We assume that the unaccounted-for distortion is purely radial
        #
        applyDistortion = self.applyDistortion
        try:
            self.applyDistortion = False
            for i in range(niter):
                # reverse transform, i.e. back to pfi
                nx, ny = self.mcsToPfi(xyout[0], xyout[1])

                xyout[0] -= self.mcs_boresight_x_pix   # relative to centre of mcs
                xyout[1] -= self.mcs_boresight_y_pix

                nr = np.hypot(nx, ny)
                r = np.hypot(xm, ym)
                scale = 1 + lam*np.where(r < 1e-5, 0, (nr - r)/r)

                xyout[0] /= scale
                xyout[1] /= scale

                xyout[0] += self.mcs_boresight_x_pix   # include centre of mcs
                xyout[1] += self.mcs_boresight_y_pix
        finally:
            self.applyDistortion = applyDistortion

        return xyout[0], xyout[1]

    def _plotMatches(self, fig, x_fid_mm, y_fid_mm, xd, yd, fid, matchRadius, nMatch):
        if fig is None:
            return

        ax = fig.gca()

        for x, y in zip(x_fid_mm, y_fid_mm):
            c = Circle((x, y), matchRadius, color='red', alpha=0.5)
            ax.add_patch(c)
        #plt.plot(x_fid_mm, y_fid_mm, 'o', label="fiducials")
        plt.plot(xd, yd, '.', label="detections")
        ptype = '+' if plt.gca().get_legend() is None else 'x'
        plt.plot(xd[fid > 0], yd[fid > 0], ptype, color='black', label=f"{nMatch} matches",
                 markersize=4*(1 if ptype=='x' else np.sqrt(2)), zorder=10)

        handles, labels = ax.get_legend_handles_labels()
        if True:
            handles += [c]
            labels += ["search"]
        plt.legend(handles, labels, handler_map={Circle: CircleHandler()})
        plt.title(f"Matched {nMatch} points")
        plt.gca().set_aspect('equal')

class SimpleTransform(PfiTransform):
    def __init__(self, altitude=90, insrot=0, applyDistortion=True, nsigma=None, alphaRot=0):
        self.setParams(altitude, insrot, nsigma, alphaRot)

        self.applyDistortion = applyDistortion

    def setParams(self, altitude=90, insrot=0, nsigma=None, alphaRot=0):
        self.altitude = altitude
        self.insrot = insrot
        self.nsigma = nsigma
        self.alphaRot = alphaRot
        #
        # The correct number of initial values; must match code in __call__()
        #
        self.mcsDistort = MeasureDistortion([], [], [], [], [], [], nsigma=0, alphaRot=self.alphaRot)
        self.mcsDistort.setArgs(np.array([0, 0, -insrot, 0, 0], dtype=float))

    def updateTransform(self, mcs_data, fiducials, matchRadius=1, nMatchMin=0.75, fig=None):
        """Update our estimate of the transform, based on the positions of fiducial fibres

        mcs_data:              Pandas DataFrame containing mcs_center_x_pix, mcs_center_y_pix
                               (measured positions in pixels in metrology camera)
        fiducials:             Pandas DataFrame containing x_mm, y_mm, fiducialId
                               (Fiducial positions in mm on PFI)
                               As returned by `butler.get("fiducials")
        matchRadius:           Radius to match points and fiducials (mm)
        nMatchMin:             Minimum number of permissible matches
                               (if <= 1, interpreted as the fraction of the number of fiducials)
        fig                    matplotlib figure for displays; or None

        Returns:
           fids:      array of `int`
                array of length mcs_data giving indices into fiducials or -1
           distance:  array of `float`
                array of length mcs_data giving distance in mm to nearest fiducial
        """
        if nMatchMin <= 1:
            nMatchMin *= len(fiducials.fiducialId)

        mcs_x_pix = mcs_data.mcs_center_x_pix.to_numpy()
        mcs_y_pix = mcs_data.mcs_center_y_pix.to_numpy()

        x_fid_mm = fiducials.x_mm.to_numpy()
        y_fid_mm = fiducials.y_mm.to_numpy()
        fiducialId = fiducials.fiducialId.to_numpy()

        xd, yd = self.mcsToPfi(mcs_x_pix, mcs_y_pix)
        fid, dmin = matchIds(xd, yd, x_fid_mm, y_fid_mm, fiducialId, matchRadius=matchRadius)
        nMatch = sum(fid > 0)

        self._plotMatches(fig, x_fid_mm, y_fid_mm, xd, yd, fid, matchRadius, nMatch)

        if nMatch < nMatchMin:
            raise RuntimeError(f"I only matched {nMatch} out of {len(fiducialId)} fiducial fibres")

        distortion = MeasureDistortion(mcs_x_pix, mcs_y_pix, fid, x_fid_mm, y_fid_mm, fiducialId,
                                       self.nsigma, self.alphaRot)
        distortion.frozen = self.mcsDistort.frozen

        res = scipy.optimize.minimize(distortion, distortion.getArgs(), method='Powell')
        self.mcsDistort.setArgs(res.x)

        xd, yd = self.mcsToPfi(mcs_x_pix, mcs_y_pix)
        fid, dmin = matchIds(xd, yd, x_fid_mm, y_fid_mm, fiducialId, matchRadius=matchRadius)

        self._plotMatches(fig, x_fid_mm, y_fid_mm, xd, yd, fid, matchRadius, nMatch)

        return (fid, dmin)

    def mcsToPfi(self, x, y):
        """transform camera pixels to pfi mm
        x, y:  position in mcs pixels

        returns:
            x, y in pfi mm
        """
        if isinstance(x, pd.Series):
            x = x.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        return self.mcsDistort.distort(x, y, inverse=False)

    def pfiToMcs(self, x, y, niter=5, lam=1.0):
        """transform pfi mm to camera mcs pixels
        x, y:  position in pfi mm
        niter: number of iterations (ignored)
        lam:   convergence factor for iteration (ignored)

        returns:
            x, y in mcs pixels
        """
        if isinstance(x, pd.Series):
            x = x.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        return self.mcsDistort.distort(x, y, inverse=True)


class ASRD71MTransform(SimpleTransform):
    """A version of SimpleTransform that's initialised for the ASRD RMOD 71M"""
    def setParams(self, altitude=90, insrot=0, nsigma=None, alphaRot=0):
        super().setParams(altitude, insrot, nsigma, alphaRot)

        self.mcsDistort.setArgs([-376.0, -268.71, -0.575, -0.924753269, -2.25647580e-13])

class USMCSTransform(SimpleTransform):
    """A version of SimpleTransform that's initialised for the ASRD RMOD 71M"""
    def setParams(self, altitude=90, insrot=0, nsigma=None, alphaRot=0):
        super().setParams(altitude, insrot, nsigma, alphaRot)

        self.mcsDistort.setArgs([-240, 350, insrot, -0.929, -2.25647580e-13])
