from pfs.utils.database.db import DB


class QaDB(DB):
    """Quality Assurance database convenience subclass of DB.

    See Also
    --------
    pfs.utils.database.db.DB
        The base class that implements connection and query helpers.
    """

    # Default parameters for the QA database
    host = "pfsa-db"
    user = "pfs"
    dbname = "qadb"
    port = 5436
