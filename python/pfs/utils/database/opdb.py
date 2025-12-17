from pfs.utils.database.db import DB

class OpDB(DB):
    """Operational database convenience subclass of DB.

    See Also
    --------
    pfs.utils.database.db.DB
        The base class that implements connection and query helpers.
    """

    DEFAULT_HOST = "db-ics"
    DEFAULT_USER = "pfs"
    DEFAULT_DBNAME = "opdb"
    DEFAULT_PORT = 5432

    def __init__(self, **kwargs):
        """Initialize the OpDB instance with default connection parameters.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the base class initializer.
        """
        super().__init__(
            **kwargs
        )
