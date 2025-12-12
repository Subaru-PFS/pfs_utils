#!/usr/bin/env python

import datetime
import pandas as pd
from pfs.utils.database.opdb import OpDB

__all__ = ["ingestCobraGeometry"]

def getXmlInfo():
    from pfs.utils.butler import Butler
    butler = Butler()
    pfi = butler.get('moduleXml', moduleName='ALL', version='')
    return pfi

def ingestCobraGeometry(calibrated_at=None, comments=None):
    """
        Description
        -----------
            Ingest the latest cobra geometry from XML file into opDB

        Parameters
        ----------
            calibrated_at : `datetime`
            comments: `str`

        Returns
        -------
            None
    """
    db = OpDB()

    # get cobra information from XML file
    pfi = getXmlInfo()

    # populate `cobra_geometry_calib`
    if calibrated_at is None:
        calibrated_at = datetime.datetime.now()
    df = pd.DataFrame({
        'cobra_geometry_calib_id': [None],
        'calibrated_at': [calibrated_at],
        'comments': [comments],
    })
    db.insert_dataframe('cobra_geometry_calib', df=df)

    # get cobra_geometry_calib_id for the latest one
    cobra_geometry_calib_id = db.query_scalar(
        'SELECT max(cobra_geometry_calib_id) AS cobra_geometry_calib_id FROM cobra_geometry_calib'
    )

    # populate `cobra_geometry`
    df = pd.DataFrame({
        'cobra_geometry_calib_id': [cobra_geometry_calib_id] * pfi.nCobras,
        'cobra_id': [i + 1 for i in range(pfi.nCobras)],
        'center_x_mm': pfi.centers.real,
        'center_y_mm': pfi.centers.imag,
        'motor_theta_limit0': pfi.tht0,
        'motor_theta_limit1': pfi.tht1,
        'motor_theta_length_mm': pfi.L1,
        'motor_phi_limit_in': pfi.phiIn,
        'motor_phi_limit_out': pfi.phiOut,
        'motor_phi_length_mm': pfi.L2,
        'status': pfi.status,
    })
    db.insert_dataframe('cobra_geometry', df=df)

if __name__ == '__main__':
    ingestCobraGeometry()
