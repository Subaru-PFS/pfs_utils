import numpy as np
import pandas as pd
import scipy.optimize

__all__ = ["PfiTransform"]

from .CoordTransp import CoordinateTransform


def matchIds(u, v, x, y, fid, matchRadius=2):
    """Given a set of points (x, y, fid) which define the ids for the points (x, y)
    and a set of points (u, v), return an array of the ids for (u, v)
    """
    fid_out = np.full_like(u, -1, dtype=fid.dtype)
    for i, (up, vp) in enumerate(zip(u, v)):
        d = np.hypot(x - up, y - vp)
        if min(d) < matchRadius:
            fid_out[i] = fid[np.argmin(d)]

    return fid_out


class MeasureDistortion:
    def __init__(self, x, y, fid, x_mm, y_mm, fiducialId):
        """x, y: measured positions in pfi coordinates
        fid: fiducial ids for (x, y) (-1: unidentified)
        x_mm, y_mm : true positions in pfi coordinates
        fiducialId: fiducial ids for (x_mm, y_mm)
        """
        _x = []
        _y = []
        _x_mm = []
        _y_mm = []
        #
        # The correct number of initial values; must match code in __call__()
        #
        x0, y0, dscale, theta, scale2 = np.array([0, 0, 0, 0, 0], dtype=float)
        self._args = np.array([x0, y0, dscale, theta, scale2])
        self.freeze = np.zeros(len(self._args), dtype=bool)

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

    def __call__(self, args):
        tx, ty = self.distort(self.x, self.y, *args)

        return np.mean(np.hypot(tx - self.x_mm, ty - self.y_mm)**2)

    def getArgs(self):
        return self._args

    def setArgs(self, *args):
        self._args[~self.freeze] = np.array(args[0])[~self.freeze]

    def distort(self, x, y, *args, **kwargs):
        if args:
            args = np.array(args)
        else:
            args = self._args

        args[self.freeze] = self._args[self.freeze]

        x0, y0, dscale, theta, scale2 = args  # must match length of self._args in __init__
        inverse = kwargs.get("inverse", False)

        theta = np.deg2rad(theta)
        c, s = np.cos(theta), np.sin(theta)

        r = np.hypot(x - x0, y - y0) if inverse else np.hypot(x, y)
        scale = (1 + dscale) + scale2*r**2

        if inverse:
            x -= x0
            y -= y0
            s = -s    # change sign of theta
            tx = ( c*x + s*y)/scale
            ty = (-s*x + c*y)/scale
        else:
            tx = x0 + scale*( c*x + s*y)
            ty = y0 + scale*(-s*x + c*y)

        return tx, ty


class PfiTransform:
    def __init__(self, altitude=90, insrot=0, applyDistortion=True):
        self.setParams(altitude, insrot)

        self.applyDistortion = applyDistortion

    def setParams(self, altitude=90, insrot=0):
        self.altitude = altitude
        self.insrot = insrot
        #
        # Values from Jennifer
        #
        if abs(altitude - 90) < 1:
            self.mcs_boresight_x_pix = 4468.6
            self.mcs_boresight_y_pix = 2869.6
        elif abs(altitude - 60) < 1:
            self.mcs_boresight_x_pix = 4466.9
            self.mcs_boresight_y_pix = 2867.8
        elif abs(altitude - 30) < 1:
            self.mcs_boresight_x_pix = 4474.9
            self.mcs_boresight_y_pix = 2849.9
        else:
            raise RuntimeError(f"I'm afraid that I can't handle altitude={altitude}")
        #
        # Initial camera distortion; updated using updateTransform
        #
        self.mcsDistort = MeasureDistortion([], [], [], [], [], [])

        a, b, phi = [1.882, 1.000], 0.0703, np.full(2, 32.2) + [90, 0]
        x0, y0 = a + b*np.sin(np.deg2rad(insrot - phi))
        dscale = 0.00266
        theta = 0.708
        scale2 = 2.38e-09
        args = [x0, y0, dscale, theta, scale2]

        self.mcsDistort.setArgs(args)

    def updateTransform(self, mcs_x_pix, mcs_y_pix, x_fid_mm, y_fid_mm, fiducialId, matchRadius=1, nMatchMin=0.75):
        """Update our estimate of the transform, based on the positions of fiducial fibres

        mcs_x_pix, mcs_y_pix:  Measured positions in pixels in metrology camera
        x_fid_mm, y_fid_mm:    Fiducial positions in mm on PFI
        matchRadius:           Radius to match points and fiducials (mm)
        nMatchMin:             Minimum number of permissible matches
                               (if <= 1, interpreted as the fraction of the number of fiducials)
        """
        if nMatchMin <= 1:
            nMatchMin *= len(fiducialId)

        if isinstance(mcs_x_pix, pd.Series):
            mcs_x_pix = mcs_x_pix.to_numpy()
        if isinstance(mcs_y_pix, pd.Series):
            mcs_y_pix = mcs_y_pix.to_numpy()
        if isinstance(x_fid_mm, pd.Series):
            x_fid_mm = x_fid_mm.to_numpy()
        if isinstance(y_fid_mm, pd.Series):
            y_fid_mm = y_fid_mm.to_numpy()
        if isinstance(fiducialId, pd.Series):
            fiducialId = fiducialId.to_numpy()

        # Get our best estimate of the transformed positions to give ourselves the
        # best chance of matching to the fiducial fibres
        ptd = PfiTransform(insrot=self.insrot, altitude=self.altitude, applyDistortion=True)
        xd, yd = ptd.mcsToPfi(mcs_x_pix, mcs_y_pix)
        del ptd

        fid = matchIds(xd, yd, x_fid_mm, y_fid_mm, fiducialId, matchRadius=matchRadius)
        nMatch = sum(fid > 0)
        if nMatch < nMatchMin:
            raise RuntimeError(f"I only matched {nMatch} out of {len(fiducialId)} fiducial fibres")

        applyDistortion = self.applyDistortion
        try:
            self.applyDistortion = False
            x, y = self.mcsToPfi(mcs_x_pix, mcs_y_pix)
        finally:
            self.applyDistortion = applyDistortion

        distortion = MeasureDistortion(x, y, fid, x_fid_mm, y_fid_mm, fiducialId)

        res = scipy.optimize.minimize(distortion, distortion.getArgs(), method='Powell')
        self.mcsDistort.setArgs(res.x)

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

        x, y = x.copy(), y.copy()
        xyin = np.stack((x, y))

        xyin[0] = -(xyin[0] - self.mcs_boresight_x_pix)   # rotate 180 about centre
        xyin[1] = -(xyin[1] - self.mcs_boresight_y_pix)

        xy = CoordinateTransform(xyin, 90.0 - self.altitude, "mcs_pfi", inr=self.insrot)
        x, y = xy[0], xy[1]

        if self.applyDistortion:
            x, y = self.mcsDistort.distort(x, y, inverse=False)

        return x, y

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

        x, y = x.copy(), y.copy()
        if self.applyDistortion:
            x, y = self.mcsDistort.distort(x, y, inverse=True)

        xyin = np.stack((x, y))
        xyout = CoordinateTransform(xyin, 90.0 - self.altitude, "pfi_mcs", inr=self.insrot)  # first guess at MCS coordinates
        #
        # Apparently there's a missing rotation by pi/2 in the pfs_utils code
        #
        tmp = xyout.copy()
        xyout[0], xyout[1] = tmp[1], -tmp[0]
        del tmp
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
                r = np.hypot(x, y)
                scale = 1 + lam*np.where(r < 1e-5, 0, (nr - r)/r)

                xyout[0] /= scale
                xyout[1] /= scale

                xyout[0] += self.mcs_boresight_x_pix   # include centre of mcs
                xyout[1] += self.mcs_boresight_y_pix
        finally:
            self.applyDistortion = applyDistortion

        return xyout[0], xyout[1]
