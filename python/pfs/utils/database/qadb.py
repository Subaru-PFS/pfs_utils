from pfs.utils.database.db import DB


class QaDB(DB):
    """Quality Assurance database convenience subclass of DB.

    See Also
    --------
    pfs.utils.database.db.DB
        The base class that implements connection and query helpers.
    """

    DEFAULT_HOST = "pfsa-db"
    DEFAULT_USER = "pfs"
    DEFAULT_DBNAME = "qadb"
    DEFAULT_PORT = 5436
