from pfs.utils.database.db import DB


class OpDB(DB):
    """Operational database convenience subclass of DB.

    See Also
    --------
    pfs.utils.database.db.DB
        The base class that implements connection and query helpers.
    """

    # Default parameters for the operational database
    host = "db-ics"
    user = "pfs"
    dbname = "opdb"
    port = 5432
