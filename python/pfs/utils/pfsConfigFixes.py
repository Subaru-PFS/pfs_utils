from __future__ import annotations

import numpy as np

from pfs.datamodel.pfsConfig import PfsConfig, FiberStatus


__all__ = ("fixPfsConfig",)


def fixPfsConfig(pfsConfig: PfsConfig) -> None:
    """Fix a PfsConfig object.

    Parameters
    ----------
    pfsConfig : `pfs.datamodel.PfsConfig`
        The PfsConfig object to fix. The object is modified in place.
    """
    if pfsConfig.visit in range(144587, 145150):
        fixRun29(pfsConfig)
        return


def fixRun29(pfsConfig: PfsConfig) -> None:
    """Fix pfsConfig for run 29.

    During Run 29, two MTPs are faint:

    SM1, fiberId= 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110
        111 112 113 114 115 116 117 118 119 120 121 122 (U2-1-1 group)
    SM4, fiberId= 2337 2338 2339 2340 2341 2342 2343 2344 2345 2346 2347 2348
        2349 2350 2351 2352 2353 2354 2355 2356 2357 2358 2359 2360 2361 2362
        2363 2364 (D1-1-4 group)

    Parameters
    ----------
    pfsConfig : `pfs.datamodel.PfsConfig`
        The PfsConfig object to fix. The object is modified in place.
    """
    fiberId = [
        94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
        109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
        2337, 2338, 2339, 2340, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348,
        2349, 2350, 2351, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360,
        2361, 2362, 2363, 2364,
    ]
    select = np.isin(pfsConfig.fiberId, fiberId)
    pfsConfig.fiberStatus[select] = FiberStatus.BLOCKED
