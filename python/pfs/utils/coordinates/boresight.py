"""Measure the MCS boresight: the point, in MCS pixels, the instrument rotator turns the focal
plane about.

Every cobra and fiducial identified in a set of MCS frames taken at several rotator angles is a
fixed point of the focal plane, so its spot in frame k is

    p_jk = c + R(theta_k) d_j

where c is the boresight, theta_k the rotation of frame k and d_j the point's offset from c.
c and every theta_k but one are fitted; d_j is solved for in closed form. The rotations are
fitted rather than taken from the rotator angle (insrot), which is good to a few tenths of a
degree only; the image turns by -insrot, which seeds them.

The error of c is dominated by a displacement common to each frame (seeing, vibration), not by
centroiding. A frame displaced far more than the others is left out; the uncertainty of c is the
error that the displacement measured in the remaining frames leads to for their rotator
positions. The boresight depends on the telescope altitude, so one measurement takes frames at a
single altitude.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

__all__ = ['BoresightResult', 'measureBoresight']

SAME_POSITION = 10
"""Frames whose rotations are closer than this, degrees, count as one rotator position: a pair
this close carries too little rotation to measure a centre."""

MAX_ALTITUDE_SPREAD = 0.5
"""Widest spread of telescope altitude, degrees, among the frames of one measurement."""

FRAME_NOISE = 0.07
"""Floor of the displacement common to a frame, pixels rms per axis: the smallest seen, on visit
153951. Below three rotator positions the displacement cannot be measured and this is used."""

DISPLACEMENT_SCALE = 0.2
"""Fixed scale, pixels per axis, a frame's displacement is judged against: generous enough for a
mediocre night, and not taken from the frames themselves, where a displaced frame would hide."""

FRAME_SHIFT_CLIP = 5
"""A frame displaced from where the other frames put it by more than this many times
DISPLACEMENT_SCALE (combined with the uncertainty of that prediction) is left out."""

POINT_CLIP = 5
"""A spot whose residual exceeds this many times 1.4826 x the median residual of the spots is
left out of the fit."""

MAX_CLIP_ITERATIONS = 5
"""Most rounds of spot clipping in one fit; the spots the last round would clip stay in."""

MAX_REJECTED_FRAMES = 2
"""Most frames left out of one measurement."""

MAX_RESIDUAL = 1
"""Largest median spot residual, pixels, of a measurement: above it the frames are not one rigid
rotation."""

MAX_ROTATION_OFFSET = 1
"""Largest difference, degrees, between a frame's fitted rotation and -insrot, relative to the
median frame, before warning."""


@dataclass
class BoresightResult:
    """A boresight measured from frames at several rotator angles.

    Attributes
    ----------
    x, y : `float`
        The boresight, MCS pixels.
    xErr, yErr : `float`
        Its uncertainty within the measurement, pixels: the error ``sigmaF`` leads to for the
        rotator positions used.
    sigmaF : `float`
        Displacement common to a rotator position, pixels rms per axis: measured from the
        positions used when there are three or more, never below FRAME_NOISE.
    sigmaFMeasured : `bool`
        Whether ``sigmaF`` was measured, rather than FRAME_NOISE taken.
    altitude : `float`
        Mean telescope altitude of the frames used, degrees.
    frames : `pandas.DataFrame`
        One row per frame given, in the order given: ``mcs_frame_id``, ``insrot``,
        ``altitude``, ``nPoints`` (spots identified), ``nLeftOut`` (spots of the frame not fitted:
        clipped, or of a point seen in this frame only), ``residual`` (median spot residual,
        pixels), ``displacement`` (how far the frame is displaced as a whole from where the other
        frames put it, pixels; NaN when it could not be tested), ``rotation`` (fitted, degrees,
        relative to the first frame used), ``rotationOffset`` (fitted rotation minus -insrot,
        degrees, relative to the median frame), ``position`` (rotator position index), ``used``
        and ``reason`` (why a frame was left out, or ''). The columns of a frame left out are
        those of the fit that left it out.
    nPositions : `int`
        Rotator positions among the frames used.
    nPoints, nLeftOut : `int`
        Spot measurements of the frames used that were fitted, and that were not.
    standingOut : `int` or `None`
        mcs_frame_id of a frame still displaced once MAX_REJECTED_FRAMES were left out.
    warnings : `list` of `str`
        What a reader of the measurement should know.
    """

    x: float
    y: float
    xErr: float
    yErr: float
    sigmaF: float
    sigmaFMeasured: bool
    altitude: float
    frames: pd.DataFrame
    nPositions: int
    nPoints: int
    nLeftOut: int
    standingOut: int = None
    warnings: list = field(default_factory=list)

    @property
    def framesUsed(self):
        """mcs_frame_id of the frames the boresight was measured from."""
        return self.frames.mcs_frame_id[self.frames.used].tolist()


def angleBetween(a, b):
    """Absolute difference of two angles in degrees, wrapped to [0, 180]."""
    return np.abs((np.asarray(a) - b + 180) % 360 - 180)


def rotatorPositions(angles):
    """Group frames by rotator position.

    Parameters
    ----------
    angles : sequence of `float`
        Rotation of each frame, degrees.

    Returns
    -------
    `numpy.ndarray` of `int`
        A position index per frame; frames within SAME_POSITION of a position's first frame
        share it.
    """
    positions = []
    index = np.empty(len(angles), dtype=int)
    for i, angle in enumerate(angles):
        for j, position in enumerate(positions):
            if angleBetween(angle, position) < SAME_POSITION:
                index[i] = j
                break
        else:
            index[i] = len(positions)
            positions.append(angle)
    return index


def checkFrames(frames):
    """Refuse frames that cannot make one measurement.

    Parameters
    ----------
    frames : `pandas.DataFrame`
        ``mcs_frame_id``, ``insrot`` and ``altitude``, degrees.

    Raises
    ------
    ValueError
        An angle is not finite, the altitudes spread over more than MAX_ALTITUDE_SPREAD, or the
        frames are at a single rotator position.
    """
    frameIds = frames.mcs_frame_id.tolist()
    insRot = frames.insrot.to_numpy(dtype=float)
    altitude = frames.altitude.to_numpy(dtype=float)

    if not (np.isfinite(insRot).all() and np.isfinite(altitude).all()):
        raise ValueError(
            f'frames {frameIds} have an unknown insrot or altitude: insrot {insRot.tolist()}, '
            f'altitude {altitude.tolist()}'
        )
    if altitude.max() - altitude.min() > MAX_ALTITUDE_SPREAD:
        raise ValueError(
            f'frames {frameIds} span altitudes {altitude.min():.2f} to {altitude.max():.2f} deg: '
            f'the boresight depends on altitude, measure one altitude at a time'
        )
    if rotatorPositions(-insRot).max() < 1:
        raise ValueError(
            f'frames {frameIds} are at a single rotator position (insrot {insRot.tolist()}): '
            f'the boresight needs frames at least {SAME_POSITION} deg apart'
        )


def rotation(theta):
    """Rotation matrices for the angles theta (radians), shape ``theta.shape + (2, 2)``."""
    cos, sin = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([cos, -sin], axis=-1), np.stack([sin, cos], axis=-1)], axis=-2)


def template(centre, theta, frame, point, xy):
    """Least-squares d_j of p_jk = c + R(theta_k) d_j, for c and theta given.

    Parameters
    ----------
    centre : `numpy.ndarray` (2,)
    theta : `numpy.ndarray` (nFrames,)
        Rotation of each frame, radians.
    frame, point : `numpy.ndarray` of `int` (n,)
        Frame and point of each measurement; point runs over 0..m-1, each seen at least once.
    xy : `numpy.ndarray` (n, 2)
        Measured spots, pixels.

    Returns
    -------
    `numpy.ndarray` (m, 2)
        Each point's offset from c at rotation 0, pixels: the mean of its measurements rotated
        back.
    """
    # each measurement rotated back: R^T (p - c).
    local = np.einsum('nji,nj->ni', rotation(theta)[frame], xy - centre)
    count = np.bincount(point)
    return np.stack([np.bincount(point, weights=local[:, i]) for i in range(2)], axis=-1) / count[:, None]


def residuals(centre, theta, frame, point, xy):
    """Residuals of p_jk = c + R(theta_k) d_j with d_j solved for: (n, 2) pixels.

    Arguments as `template`.
    """
    d = template(centre, theta, frame, point, xy)
    return xy - centre - np.einsum('nij,nj->ni', rotation(theta)[frame], d[point])


def seedCentre(theta, frame, point, xy):
    """Linear estimate of c with the rotations held, against the frame with the most spots.

    Solves (I - R_k) c = p_jk - R_k p_jr, R_k the rotation from the reference frame r to frame k,
    in least squares over every spot also seen in the reference frame. Good to a fraction of a
    pixel with rotations from insrot, which is all the fit needs to start from.

    Parameters
    ----------
    theta : `numpy.ndarray` (nFrames,)
        Rotation of each frame, radians.
    frame, point : `numpy.ndarray` of `int` (n,)
        Frame and point of each measurement.
    xy : `numpy.ndarray` (n, 2)
        Measured spots, pixels.

    Returns
    -------
    `numpy.ndarray` (2,)
        Pixels.

    Raises
    ------
    ValueError
        No spot of the reference frame is seen in a frame at another rotation.
    """
    reference = np.argmax(np.bincount(frame, minlength=len(theta)))
    known = np.full((point.max() + 1, 2), np.nan)
    known[point[frame == reference]] = xy[frame == reference]
    use = (frame != reference) & np.isfinite(known[point, 0])

    R = rotation(theta[frame[use]] - theta[reference])
    A = np.eye(2) - R
    normal = np.einsum('nki,nkj->ij', A, A)
    if not use.any() or np.linalg.cond(normal) > 1e8:
        raise ValueError('no spot of the frame with the most spots is seen in a frame at another rotation')
    rhs = xy[use] - np.einsum('nij,nj->ni', R, known[point[use]])
    return np.linalg.solve(normal, np.einsum('nki,nk->i', A, rhs))


def fitCentre(theta, frame, point, xy, centre):
    """Fit c and every rotation but the first to the measurements.

    Parameters
    ----------
    theta : `numpy.ndarray` (nFrames,)
        Starting rotations, radians; theta[0] is held.
    frame, point : `numpy.ndarray` of `int` (n,)
        Frame and point of each measurement; point runs over 0..m-1.
    xy : `numpy.ndarray` (n, 2)
        Measured spots, pixels.
    centre : `numpy.ndarray` (2,)
        Starting centre, pixels.

    Returns
    -------
    centre : `numpy.ndarray` (2,)
    theta : `numpy.ndarray` (nFrames,)
        Fitted rotations, radians.
    residual : `numpy.ndarray` (n, 2)

    Raises
    ------
    RuntimeError
        The fit did not converge.
    """

    def cost(params):
        return residuals(params[:2], np.concatenate([theta[:1], params[2:]]), frame, point, xy).ravel()

    # MINPACK's lm rather than trf, whose dense linear algebra runs threaded BLAS that stalls for
    # minutes in a process which also loads cv2.
    fit = optimize.least_squares(cost, np.concatenate([centre, theta[1:]]), method='lm', x_scale='jac')
    if not fit.success:
        raise RuntimeError(f'boresight fit did not converge: {fit.message}')

    theta = np.concatenate([theta[:1], fit.x[2:]])
    return fit.x[:2], theta, residuals(fit.x[:2], theta, frame, point, xy)


@dataclass
class FrameFit:
    """The fit of a set of frames: see `fitFrames`."""

    centre: np.ndarray
    theta: np.ndarray
    residual: np.ndarray
    used: np.ndarray
    offsets: np.ndarray


def fitFrames(insRot, frame, point, xy, *, nPoints):
    """Fit the boresight to the spots of a set of frames, clipping outlying spots.

    Parameters
    ----------
    insRot : `numpy.ndarray` (nFrames,)
        Rotator angle of each frame, degrees; seeds its rotation.
    frame, point : `numpy.ndarray` of `int` (n,)
        Frame (indexing insRot) and point of each measurement; point indexes the points of the
        whole measurement.
    xy : `numpy.ndarray` (n, 2)
        Measured spots, pixels.
    nPoints : `int`
        Number of points of the whole measurement, which ``offsets`` is sized to.

    Returns
    -------
    `FrameFit`
        ``centre`` (2,); ``theta`` (nFrames,), fitted rotations, radians; ``residual`` (n, 2),
        pixels, of every spot of a fitted point, clipped spots included, NaN otherwise; ``used``
        (n,) bool, the spots fitted; ``offsets`` (nPoints, 2), each fitted point's d_j,
        NaN otherwise.

    Raises
    ------
    ValueError
        A frame shares no point with the others.
    """
    theta = -np.deg2rad(insRot)
    kept = np.ones(len(xy), dtype=bool)
    centre = None

    for iteration in range(MAX_CLIP_ITERATIONS):
        seen = np.bincount(point[kept], minlength=point.max() + 1)
        used = kept & (seen[point] >= 2)
        alone = np.setdiff1d(np.arange(len(insRot)), frame[used])
        if len(alone):
            raise ValueError(f'frames {alone.tolist()} (of those fitted) share no point with the others')

        fitted, pointIndex = np.unique(point[used], return_inverse=True)
        if centre is None:
            centre = seedCentre(theta, frame[used], pointIndex, xy[used])

        centre, theta, residual = fitCentre(theta, frame[used], pointIndex, xy[used], centre)
        distance = np.hypot(*residual.T)
        clipped = distance > POINT_CLIP * 1.4826 * np.median(distance)
        if not clipped.any() or iteration == MAX_CLIP_ITERATIONS - 1:
            break
        kept[np.flatnonzero(used)[clipped]] = False

    offsets = np.full((nPoints, 2), np.nan)
    offsets[fitted] = template(centre, theta, frame[used], pointIndex, xy[used])
    has = np.isfinite(offsets[point, 0])
    full = np.full((len(xy), 2), np.nan)
    full[has] = xy[has] - centre - np.einsum('nij,nj->ni', rotation(theta)[frame[has]], offsets[point[has]])
    return FrameFit(centre=centre, theta=theta, residual=full, used=used, offsets=offsets)


def frameShift(centre, xy, d):
    """How far a frame is displaced as a whole from where a fit of other frames puts it.

    The frame's rotation is fitted in closed form against the points' offsets; the displacement
    is the median of its spots' residuals, per axis, which a few mismatches cannot move.

    Parameters
    ----------
    centre : `numpy.ndarray` (2,)
        Centre fitted to the other frames, pixels.
    xy : `numpy.ndarray` (n, 2)
        The frame's spots, pixels.
    d : `numpy.ndarray` (n, 2)
        Offsets d_j of those spots' points, fitted to the other frames, pixels.

    Returns
    -------
    `numpy.ndarray` (2,)
        Pixels.
    """
    q = xy - centre
    theta = np.arctan2(
        np.sum(d[:, 0] * q[:, 1] - d[:, 1] * q[:, 0]), np.sum(d[:, 0] * q[:, 0] + d[:, 1] * q[:, 1])
    )
    return np.median(q - d @ rotation(theta).T, axis=0)


def expectedError(theta, sigma):
    """Error of c, pixels per axis, that a displacement sigma per rotator position leads to.

    Each position's spots satisfy p = c + R(theta_k) d + t_k: in the displacements t_k the centre
    enters as b_k = c + R(theta_k) u, a linear model in c and u whose covariance gives the error
    of c.

    Parameters
    ----------
    theta : `numpy.ndarray` (nPositions,)
        Rotation of each position, radians.
    sigma : `float`
        Displacement per position, pixels rms per axis.

    Returns
    -------
    `numpy.ndarray` (2,)
    """
    X = np.concatenate([np.broadcast_to(np.eye(2), (len(theta), 2, 2)), rotation(theta)], axis=2).reshape(
        -1, 4
    )
    covariance = sigma**2 * np.linalg.inv(X.T @ X)
    return np.sqrt(np.diag(covariance)[:2])


def positionDisplacements(fit, frame, position, theta):
    """Mean residual of each rotator position's fitted spots, and the mean rotation of each.

    Parameters
    ----------
    fit : `FrameFit`
    frame : `numpy.ndarray` of `int` (n,)
        Frame of each measurement of the fit, indexing ``position`` and ``theta``.
    position : `numpy.ndarray` of `int` (nFrames,)
        Rotator position of each frame.
    theta : `numpy.ndarray` (nFrames,)
        Fitted rotation of each frame, radians.

    Returns
    -------
    displacement : `numpy.ndarray` (nPositions, 2)
        Pixels.
    positionTheta : `numpy.ndarray` (nPositions,)
        Radians.
    """
    used = fit.used
    frameMean = np.array([np.mean(fit.residual[used & (frame == k)], axis=0) for k in range(len(theta))])
    nPositions = position.max() + 1
    displacement = np.array([frameMean[position == p].mean(axis=0) for p in range(nPositions)])
    positionTheta = np.array(
        [np.angle(np.mean(np.exp(1j * theta[position == p]))) for p in range(nPositions)]
    )
    return displacement, positionTheta


def measureBoresight(frames, points):
    """Measure the boresight from frames at several rotator angles and one altitude.

    A frame displaced as a whole from where the other frames put it is left out,
    MAX_REJECTED_FRAMES at most, provided the other frames span three rotator positions or more;
    a spot whose residual stands out is left out. The result says which and why.

    Parameters
    ----------
    frames : `pandas.DataFrame`
        One row per frame: ``mcs_frame_id`` (distinct), ``insrot`` and ``altitude`` (degrees).
    points : `pandas.DataFrame`
        Identified spots: ``mcs_frame_id``, ``kind`` and ``id`` (naming the point, e.g.
        'cobra' and cobra_id, 'fiducial' and fiducial_fiber_id), ``x`` and ``y`` (MCS pixels,
        finite).

    Returns
    -------
    `BoresightResult`

    Raises
    ------
    ValueError
        The frames are of several altitudes or a single rotator position, fewer than two have
        identified spots, a frame shares no point with the others, or their spots do not fit one
        rotation within MAX_RESIDUAL.
    """
    frames = frames.reset_index(drop=True).copy()
    checkFrames(frames)

    withSpots = frames.mcs_frame_id.isin(points.mcs_frame_id)
    if withSpots.sum() < 2:
        raise ValueError(
            f'fewer than two frames with identified spots: frames '
            f'{frames.mcs_frame_id[withSpots].tolist()} of {frames.mcs_frame_id.tolist()}'
        )

    frame = pd.Series(frames.index, index=frames.mcs_frame_id)[points.mcs_frame_id].to_numpy()
    point = pd.factorize(pd.MultiIndex.from_arrays([points.kind, points.id]))[0]
    xy = points[['x', 'y']].to_numpy(dtype=float)
    insRot = frames.insrot.to_numpy(dtype=float)

    frames['nPoints'] = np.bincount(frame, minlength=len(frames))
    frames['used'] = frames.nPoints > 0
    frames['reason'] = np.where(frames.used, '', 'no identified spot')
    frames['nLeftOut'] = 0
    for column in ['residual', 'displacement', 'rotation', 'rotationOffset', 'position']:
        frames[column] = np.nan
    warnings = []
    standingOut = None

    def fit(used):
        """Fit on the frames flagged in used: their indices, the spots' mask and the fit."""
        index = np.flatnonzero(used)
        mask = np.isin(frame, index)
        try:
            result = fitFrames(
                insRot[index],
                np.searchsorted(index, frame[mask]),
                point[mask],
                xy[mask],
                nPoints=point.max() + 1,
            )
        except ValueError as e:
            raise ValueError(f'frames {frames.mcs_frame_id[index].tolist()}: {e}') from None
        return index, mask, result

    for attempt in range(MAX_REJECTED_FRAMES + 1):
        index, mask, result = fit(frames.used.to_numpy())
        position = rotatorPositions(np.rad2deg(result.theta))
        if position.max() < 1:
            raise ValueError(
                f'frames {frames.mcs_frame_id[index].tolist()} are at a single rotator position once '
                f'{(~frames.used).sum()} were left out: the boresight needs frames at least '
                f'{SAME_POSITION} deg apart'
            )

        distance = np.hypot(*result.residual.T)
        for k in index:
            inFrame = frame[mask] == k
            frames.loc[k, 'residual'] = np.nanmedian(distance[inFrame])
            frames.loc[k, 'nLeftOut'] = int((inFrame & ~result.used).sum())

        # displacement: each frame against the fit of the others, which its own displacement
        # cannot pull towards it; only when the others span three positions, below which they
        # fit any frame equally well.
        shift = np.full(len(index), np.nan)
        scale = np.full(len(index), np.nan)
        for i, k in enumerate(index):
            otherTheta = np.delete(result.theta, i)
            if rotatorPositions(np.rad2deg(otherTheta)).max() < 2:
                continue
            __, __, without = fit(np.isin(np.arange(len(frames)), np.delete(index, i)))
            known = (frame == k) & np.isfinite(without.offsets[point, 0])
            if known.any():
                shift[i] = np.hypot(*frameShift(without.centre, xy[known], without.offsets[point[known]]))
                # where the others put the frame is uncertain by up to twice their centre's error.
                prediction = 2 * np.hypot(*expectedError(without.theta, DISPLACEMENT_SCALE))
                scale[i] = FRAME_SHIFT_CLIP * np.hypot(DISPLACEMENT_SCALE, prediction)
        frames.loc[index, 'displacement'] = shift

        displaced = np.where(np.isfinite(shift), shift / scale, 0)
        worst = np.argmax(displaced)
        if displaced[worst] <= 1:
            break
        if attempt == MAX_REJECTED_FRAMES:
            standingOut = int(frames.mcs_frame_id[index[worst]])
            warnings.append(
                f'frame {standingOut} is still displaced by {shift[worst]:.2f} pix from the other '
                f'frames after leaving out {attempt}'
            )
            break
        frames.loc[index[worst], ['used', 'reason']] = (
            False,
            f'displaced by {shift[worst]:.2f} pix from the other frames',
        )

    if not np.isfinite(frames.displacement[index]).any():
        warnings.append(
            'frames could not be checked against each other (fewer than four rotator positions): '
            'a frame displaced as a whole would go unnoticed'
        )

    medianResidual = np.median(np.hypot(*result.residual[result.used].T))
    if medianResidual > MAX_RESIDUAL:
        raise ValueError(
            f'the spots of frames {frames.mcs_frame_id[index].tolist()} fit one rotation to '
            f'{medianResidual:.2f} pix only, over {MAX_RESIDUAL} pix: they are not of one rigid rotation'
        )

    theta = result.theta
    frames.loc[index, 'rotation'] = np.rad2deg(theta - theta[0])
    offset = np.rad2deg(theta) + insRot[index]
    frames.loc[index, 'rotationOffset'] = offset - np.median(offset)
    frames.loc[index, 'position'] = position
    for k in index[angleBetween(frames.rotationOffset[index], 0) > MAX_ROTATION_OFFSET]:
        warnings.append(
            f'frame {frames.mcs_frame_id[k]} turned {frames.rotationOffset[k]:+.2f} deg from what '
            f'its insrot says'
        )

    nPositions = position.max() + 1
    displacement, positionTheta = positionDisplacements(
        result, np.searchsorted(index, frame[mask]), position, theta
    )
    sigmaF, measured = FRAME_NOISE, False
    if nPositions >= 3:
        sigmaF = max(np.sqrt(np.sum(displacement**2) / (2 * nPositions - 4)), FRAME_NOISE)
        measured = True
    error = expectedError(positionTheta, sigmaF)

    used = result.used
    return BoresightResult(
        x=float(result.centre[0]),
        y=float(result.centre[1]),
        xErr=float(error[0]),
        yErr=float(error[1]),
        sigmaF=float(sigmaF),
        sigmaFMeasured=measured,
        altitude=float(frames.altitude[index].mean()),
        frames=frames,
        nPositions=int(nPositions),
        nPoints=int(used.sum()),
        nLeftOut=int(len(used) - used.sum()),
        standingOut=standingOut,
        warnings=warnings,
    )
