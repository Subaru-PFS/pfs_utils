from pfs.utils.database.db import DB

DEFAULT_HOST = "pfsa-db"
DEFAULT_USER = "pfs"
DEFAULT_DBNAME = "qadb"
DEFAULT_PORT = 5436


class QaDB(DB):
    """Quality Assurance database convenience subclass of DB.

    See Also
    --------
    pfs.utils.database.db.DB
        The base class that implements connection and query helpers.
    """

    def __init__(self, **kwargs):
        """Initialize the QaDB instance with default connection parameters.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the base class initializer.
        """
        super().__init__(
            host=kwargs.get("host", DEFAULT_HOST),
            user=kwargs.get("user", DEFAULT_USER),
            dbname=kwargs.get("dbname", DEFAULT_DBNAME),
            port=kwargs.get("port", DEFAULT_PORT),
            **kwargs
        )
