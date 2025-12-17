from pfs.utils.database import db

DEFAULT_HOST = "g2sim-cat"
DEFAULT_USER = "obsuser"
DEFAULT_DBNAME = "star_catalog"
DEFAULT_PORT = 5438


class GaiaDB(db.DB):
    """Gaia catalog database convenience subclass of DB.

    Note that this database is read-only; attempts to insert data will raise
    NotImplementedError.

    See Also
    --------
    pfs.utils.database.db.DB
        The base class that implements connection and query helpers.
    """

    def __init__(self, **kwargs):
        """Initialize the GaiaDB instance with default connection parameters.

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

    def commit(self, *args, **kwargs):
        """Raise an error indicating this is a read-only database.

        Raises
        ------
        NotImplementedError
            Always raised as this is a read-only database.
        """
        raise NotImplementedError("GaiaDB is a read-only database. Commit operations are not allowed.")

    def insert_dataframe(self, *args, **kwargs):
        """Raise an error indicating this is a read-only database.

        Raises
        ------
        NotImplementedError
            Always raised as this is a read-only database.
        """
        raise NotImplementedError("GaiaDB is a read-only database. Insert operations are not allowed.")

    def insert_kw(self, *args, **kwargs):
        """Raise an error indicating this is a read-only database.

        Raises
        ------
        NotImplementedError
            Always raised as this is a read-only database.
        """
        raise NotImplementedError("GaiaDB is a read-only database. Insert operations are not allowed.")
