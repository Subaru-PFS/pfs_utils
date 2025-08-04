import numpy as np

# sines & cosines of +90, +30, -30, -90, -150, -210 (+150) deg
_SIN60 = np.array([1, 0.5, -0.5, -1, -0.5, 0.5])
_COS60 = np.array([0, 0.8660254037844386, 0.8660254037844386, 0, -0.8660254037844386, -0.8660254037844386])

# camera 1 at 9 o'clock position (241.292, 0) mm
# camera 2 at 7 o'clock position (120.646, 208.965) mm
# camera 3 at 5 o'clock position (-120.646, 208.965) mm
# camera 4 at 3 o'clock position (-241.292, 0) mm
# camera 5 at 1 o'clock position (-120.646, -208.965) mm
# camera 6 at 11 o'clock position (120.646, -208.965) mm
# (ref. Moritani-san's memo_AGcamera_20171023.pdf)


# offsets from design positions in detector plane coordinates affixed to popt2 (mm)
# (ref. Kawanomoto-san's PFS-PFI-NAJ802201-00_AGCameraOffset.pdf)
_DXDP = - np.array([+0.405, +0.055, +0.357, -0.270, -0.444, -0.067]) \
    + np.array([-0.013, +0.012, +0.030, +0.013, -0.015, -0.024]) \
    - np.array([+0.003, -0.006, -0.008, +0.011, -0.001, -0.005]) \
    - np.array([+0.002,+0.000,-0.019,+0.000,+0.094,+0.000]) \
    - np.array([+0.000,+0.000,+0.000,+0.000,+0.000,+0.000]) \
    - np.array([+0.000,+0.000,+0.000,-0.013,+0.000,+0.000]) \
    - np.array([+0.000,+0.000,+0.000,+0.000,+0.000,+0.000])
_DYDP = - np.array([-0.668, +0.081, +0.180, +0.357, +0.138, -0.077]) \
    + np.array([+0.015, +0.025, +0.014, +0.002, +0.022, +0.004]) \
    - np.array([-0.004, +0.011, +0.001, +0.000, +0.012, -0.015]) \
    - np.array([-0.099,+0.000,-0.019,+0.000,-0.081,+0.000]) \
    - np.array([-0.234,+0.000,+0.000,+0.000,+0.000,+0.000]) \
    - np.array([+0.000,+0.000,+0.000,+0.004,+0.000,+0.000]) \
    - np.array([-0.015,+0.000,+0.000,+0.000,+0.000,+0.000])
_DTDP = np.array([-0.253368, +0.234505, +0.329449, +0.416894, +0.0589071, +0.234977]) \
    + np.array([+0.000000, +0.000000, -0.0539154, +0.000000, -0.137107, +0.000000]) \
    + np.array([+0.000000, +0.000000, +0.0000000, +0.000000, +0.000000, +0.000000])
_SINDTDP = np.sin(np.deg2rad(_DTDP))
_COSDTDP = np.cos(np.deg2rad(_DTDP))


def dp2det(icam, x_dp, y_dp):
    """
    Convert detector plane coordinates to detector coordinates.

    Convert Cartesian coordinates of points on the focal plane in the detector
    plane coordinate system to those on one of the detectors in the detector
    coordinate system.

    Parameters
    ----------
    icam : array_like
        The detector identifiers ([0, 5])
    x_dp : array_like
        The Cartesian coordinates x's of points on the focal plane in the
        detector plane coordinate system (mm)
    y_dp : array_like
        The Cartesian coordinates y's of points on the focal plane in the
        detector plane coordinate system (mm)

    Returns
    -------
    2-tuple of array_likes
        The Cartesian coordinates x's and y's of points on the specified
        detector in the detector coordinate system (pix)
    """

    ia = np.array(icam, dtype=np.int_)

    x_dp = x_dp - (- _DYDP[ia])
    y_dp = y_dp - _DXDP[ia]

    p = 0.013  # mm
    r = 241.314  # mm
    _sin = _SIN60[ia]
    _cos = _COS60[ia]
    x = _cos * x_dp - _sin * y_dp
    y = _sin * x_dp + _cos * y_dp
    _x_det = x / p
    _y_det = - (y - r) / p

    _sin = _SINDTDP[ia]
    _cos = _COSDTDP[ia]
    x_det = _cos * _x_det + _sin * _y_det
    y_det = - _sin * _x_det + _cos * _y_det
    x_det += 511.5 + 24
    y_det += 511.5 + 9

    return x_det, y_det


def det2dp(icam, x_det, y_det):
    """
    Convert detector coordinates to detector plane coordinates.

    Convert Cartesian coordinates of points on one of the detectors in the
    detector coordinate system to those on the focal plane in the detector
    plane coordinate system.

    Parameters
    ----------
    icam : array_like
        The detector identifiers ([0, 5])
    x_det : array_like
        The Cartesian coordinates x's of points on the specified detector in
        the detector coordinate system (pix)
    y_det : array_like
        The Cartesian coordinates y's of points on the specified detector in
        the detector coordinate system (pix)

    Returns
    -------
    2-tuple of array_likes
        The Cartesian coordinates x's and y's of points on the focal plane in
        the detector plane coordinate system (mm)
    """

    ia = np.array(icam, dtype=np.int_)

    _sin = _SINDTDP[ia]
    _cos = _COSDTDP[ia]
    x_det = x_det - (511.5 + 24)
    y_det = y_det - (511.5 + 9)
    _x_det = _cos * x_det - _sin * y_det
    _y_det = _sin * x_det + _cos * y_det

    p = 0.013  # mm
    r = 241.314  # mm
    _sin = _SIN60[ia]
    _cos = _COS60[ia]
    x = _x_det * p
    y = - _y_det * p + r
    x_dp = _cos * x + _sin * y
    y_dp = - _sin * x + _cos * y

    x_dp += - _DYDP[ia]
    y_dp += _DXDP[ia]

    return x_dp, y_dp
