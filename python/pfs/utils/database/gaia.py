from pfs.utils.database import db


class GaiaDB(db.DB):
    """Gaia catalog database convenience subclass of DB.

    Note that this database is read-only; attempts to insert data will raise
    NotImplementedError.

    See Also
    --------
    pfs.utils.database.db.DB
        The base class that implements connection and query helpers.
    """

    # Default host used for the Gen2 gaia database.
    host = "g2sim-cat"
    user = "obsuser"
    dbname = "star_catalog"
    port = 5438

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
