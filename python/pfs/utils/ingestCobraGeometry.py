#!/usr/bin/env python

import pandas as pd
from opdb import opdb

__all__ = ["ingestCobraGeometry"]

hostname = 'db-ics'
port = 5432
dbname = 'opdb'
username = 'pfs'

def getXmlInfo():
    from pfs.utils import butler
    butler = butler.Butler()
    pfi = butler.get('moduleXml', moduleName='ALL', version='')
    return pfi

def ingestCobraGeometry(calibrated_at=None, comments=None):
    '''
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
    '''
    db = opdb.OpDB(hostname=hostname,
                   port=port,
                   dbname=dbname,
                   username=username,
                   dialect='postgresql',
                   )

    ''' get cobra information from XML file '''
    pfi = getXmlInfo()

    ''' populate `cobra_geometry_calib` '''
    if calibrated_at == None:
        import datetime
        calibrated_at = datetime.datetime.now()
    df = pd.DataFrame({'cobra_geometry_calib_id': [None],
                       'calibrated_at': [calibrated_at],
                       'comments': [comments]})
    db.insert('cobra_geometry_calib', df)

    ''' get cobra_geometry_calib_id for the latest one '''
    df = db.fetch_all('cobra_geometry_calib')
    cobra_geometry_calib_id = max(df['cobra_geometry_calib_id'])

    ''' populate `cobra_geometry` '''
    df = pd.DataFrame({'cobra_geometry_calib_id': [cobra_geometry_calib_id for _ in range(pfi.nCobras)],
                       'cobra_id': [i+1 for i in range(pfi.nCobras)],
                       'center_x_mm': pfi.centers.real,
                       'center_y_mm': pfi.centers.imag,
                       'motor_theta_limit0': pfi.tht0,
                       'motor_theta_limit1': pfi.tht1,
                       'motor_theta_length_mm': pfi.L1,
                       'motor_phi_limit_in': pfi.phiIn,
                       'motor_phi_limit_out': pfi.phiOut,
                       'motor_phi_length_mm': pfi.L2,
                       'status': pfi.status})
    db.insert('cobra_geometry', df)

    db.close()

if __name__ == '__main__':
    ingestCobraGeometry()