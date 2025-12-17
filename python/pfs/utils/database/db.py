import logging
import re
from contextlib import contextmanager
from threading import RLock
from typing import Any, Mapping, Optional, Union

import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, text
from sqlalchemy.engine import Connection, Engine

_DB_ENGINES: dict[str, Engine] = {}
_DB_ENGINES_LOCK = RLock()

# Default connection parameters
DEFAULT_HOST = "localhost"
DEFAULT_USER = "user"
DEFAULT_DBNAME = "dbname"
DEFAULT_PORT = 5432


class DB:
    """Generic DB helper that accepts a DSN string or connection parameters.

    This class caches a SQLAlchemy Engine, which manages a pool of connections.
    Individual methods check out a connection from the pool for the duration of
    the call and then return it to the pool, ensuring proper connection reuse.
    For explicit reuse of a single connection across multiple operations, use
    the public `connection()` context manager.

    Note:
        Password is expected to be managed externally (e.g., ~/.pgpass).

    Usage examples:
        # Using keyword parameters
        db = DB(dbname='opdb', user='pfs', host='db-ics')

        # Using a DSN string
        db = DB('dbname=opdb user=pfs host=db-ics')

        # Using a mapping/dict
        db = DB({'dbname': 'opdb', 'user': 'pfs', 'host': 'db-ics'})
    """

    @classmethod
    def set_default_connection(
        cls,
        *,
        host: str | None = None,
        user: str | None = None,
        dbname: str | None = None,
        port: int | None = None,
    ) -> None:
        """Set default connection parameters for this class.

        Behavior
        ---------
        - When called on ``DB`` itself, updates the module-wide defaults used by
          all classes that don't override their own defaults.
        - When called on a subclass (e.g., ``OpDB`` or ``GaiaDB``), updates only
          that subclass's class-level defaults without affecting global defaults
          or other subclasses.

        Notes
        -----
        - This affects only new instances created after calling this method.
          Existing instances are not modified.
        - Passwords are not handled here; use your ``~/.pgpass`` or other
          external mechanisms as before.

        Parameters
        ----------
        host : str, optional
            Default database host.
        user : str, optional
            Default database user.
        dbname : str, optional
            Default database name.
        port : int, optional
            Default database port.

        Examples
        --------
        >>> DB.set_default_connection(host="db-ics", user="public_user", dbname="opdb", port=5432)
        >>> db = DB()  # will use the defaults above
        >>> from pfs.utils.database.opdb import OpDB
        >>> OpDB.set_default_connection(host="db-ics", user="pfs", dbname="opdb", port=5432)
        >>> op = OpDB()  # uses OpDB's class defaults only
        """
        # If invoked on the base DB class, update module-level defaults.
        if cls is DB:
            global DEFAULT_HOST, DEFAULT_USER, DEFAULT_DBNAME, DEFAULT_PORT
            if host is not None:
                DEFAULT_HOST = host
            if user is not None:
                DEFAULT_USER = user
            if dbname is not None:
                DEFAULT_DBNAME = dbname
            if port is not None:
                DEFAULT_PORT = int(port)
            return

        # Otherwise, update subclass-specific class attributes without touching globals.
        if host is not None:
            setattr(cls, "DEFAULT_HOST", host)
        if user is not None:
            setattr(cls, "DEFAULT_USER", user)
        if dbname is not None:
            setattr(cls, "DEFAULT_DBNAME", dbname)
        if port is not None:
            setattr(cls, "DEFAULT_PORT", int(port))

    def __init__(
        self,
        dsn: Optional[Union[str, Mapping[str, Any]]] = None,
        host: str | None = None,
        user: str | None = None,
        dbname: str | None = None,
        port: int | None = None,
    ) -> None:
        """Initialize a DB instance.
        Parameters
        ----------
        dsn : str or Mapping[str, Any], optional
            A DSN string (e.g., "dbname=opdb user=pfs host=db-ics") or a mapping
            of connection parameters. If provided, it takes precedence over host/user/dbname.
        host : str, optional
            Database host name.
        user : str, optional
            Database user name.
        dbname : str, optional
            Database name.
        port : int, optional
            Database port number, default is 5432.
        """
        # Resolve defaults with the following precedence:
        # 1) Explicit argument passed to __init__
        # 2) Class-level defaults on the concrete subclass (attributes: host, user, dbname, port)
        # 3) Module-wide defaults defined in this module
        cls = type(self)
        self.host = host if host is not None else getattr(cls, "host", DEFAULT_HOST)
        self.user = user if user is not None else getattr(cls, "user", DEFAULT_USER)
        self.dbname = dbname if dbname is not None else getattr(cls, "dbname", DEFAULT_DBNAME)
        resolved_port = port if port is not None else getattr(cls, "port", DEFAULT_PORT)
        self.port = int(resolved_port) if resolved_port is not None else None  # type: ignore[assignment]

        self.logger = logging.getLogger(f"DB-{self.dbname}")

        self._dsn: Optional[Union[str, Mapping[str, Any]]] = None
        if dsn is not None:
            self.dsn = dsn

        self.logger.debug(f"Connecting to database {self.dbname}")

    @property
    def dsn(self) -> str:
        """The DSN string or mapping used for the connection, if any."""
        if self._dsn is None:
            self._dsn = f"dbname={self.dbname} user={self.user} host={self.host} port={self.port}"

        if isinstance(self._dsn, Mapping):
            dsn_items = []
            for k in ("dbname", "user", "host", "port"):
                v = self._dsn.get(k)
                if v is not None:
                    dsn_items.append(f"{k}={v}")
            self._dsn = " ".join(dsn_items)

        return self._dsn

    @dsn.setter
    def dsn(self, value: Union[str, Mapping[str, Any]]) -> None:
        """Set the DSN string or mapping used for the connection."""
        self._dsn = value

        # Parse a libpq-style DSN string to update attributes
        if isinstance(value, str):
            for k, v in (item.split("=") for item in value.split() if "=" in item):
                if k == "host":
                    self.host = v
                elif k == "user":
                    self.user = v
                elif k == "dbname":
                    self.dbname = v
                elif k == "port":
                    self.port = int(v)
        elif isinstance(value, Mapping):
            if "host" in value:
                self.host = value["host"]
            if "user" in value:
                self.user = value["user"]
            if "dbname" in value:
                self.dbname = value["dbname"]
            if "port" in value:
                self.port = int(value["port"])  # type: ignore[arg-type]

    @property
    def url(self) -> str:
        """Build a SQLAlchemy URL for PostgreSQL using current attributes.

        We intentionally omit the password; libpq will use ~/.pgpass if available.
        """
        user = self.user or ""
        host = self.host or ""
        dbname = self.dbname or ""
        port = f":{self.port}" if self.port is not None else ""
        # Use psycopg (psycopg3) driver via SQLAlchemy.
        return f"postgresql+psycopg://{user}@{host}{port}/{dbname}"

    @property
    def engine(self) -> Engine | None:
        """Create or return a SQLAlchemy Engine cached via its URL."""
        with _DB_ENGINES_LOCK:
            eng = _DB_ENGINES.get(self.url)
            if eng is None:
                eng = create_engine(self.url, pool_pre_ping=True, future=True)
                _DB_ENGINES[self.url] = eng

            return eng

    def connect(self) -> Connection:
        """Return a new SQLAlchemy connection using provided DSN/params.

        Notes:
            Connections are provided by SQLAlchemy's connection pool managed by the
            cached Engine. Each call checks out a connection from the pool and, when
            closed, returns it to the pool for reuse.
        """
        return self.engine.connect()

    @contextmanager
    def connection(self):
        """Public context manager yielding a pooled SQLAlchemy connection that auto-commits.

        Use this to explicitly reuse the same connection across multiple statements:

            with db.connection() as conn:
                conn.exec_driver_sql("SELECT 1")
                conn.exec_driver_sql("SELECT 2")

        This reduces per-call checkout/return overhead while still leveraging
        SQLAlchemy's pooling. The underlying cursor lifecycle is managed by the
        DBAPI driver (psycopg3) per execution.
        """
        with self.connect() as conn:
            yield conn
            conn.commit()

    def commit(self, query: str, params: dict | list | None = None):
        """Execute a query within a transaction and commit it."""
        with self.connection() as conn:
            conn.execute(text(query), params)  # type: ignore[arg-type]

    def _execute(self, sql: str, *, params: dict | list | None, conn: Connection | None):
        """Execute SQL via native driver first, falling back to SQLAlchemy text()."""
        if re.search(r'%\(\w+\)s', sql):
            # distinct from standard %s param style, specifically looks for dictionary keys
            # Naive conversion: %(foo)s -> :foo
            sql = re.sub(r'%\((\w+)\)s', r':\1', sql)

        if conn is not None:
            return conn.execute(text(sql), params)
        with self.connection() as c:
            return c.execute(text(sql), params)

    def query(self, *args, **kwargs):
        """Thin wrapper around `query_dataframe`.

        Notes
        -----
        This method simply forwards all arguments to `query_dataframe` and
        returns its result. It exists for backward compatibility with code
        that previously used `query` to obtain a pandas DataFrame.

        Parameters
        ----------
        *args, **kwargs
            Passed directly to `query_dataframe(sql, params=None, conn=None)`.

        Returns
        -------
        pandas.DataFrame
            The query result as a DataFrame.

        Examples
        --------
        Basic usage returning a DataFrame:

        >>> db = DB()
        >>> df = db.query("SELECT 1 AS value")
        >>> int(df.loc[0, "value"]) == 1
        True

        Using parameters (named style):

        >>> df = db.query("SELECT :x + :y AS s", params={"x": 2, "y": 3})
        >>> int(df.loc[0, "s"]) == 5
        True
        """
        # Back-compat thin wrapper should return the DataFrame
        return self.query_dataframe(*args, **kwargs)

    def query_dataframe(self, sql: str, /, *, params: dict | list | None = None, conn: Connection | None = None
                        ) -> pd.DataFrame:
        """Run a SQL query and return the result as a pandas DataFrame.

        Parameters
        ----------
        sql : str
            SQL string. Named parameters are supported using the ``:name`` style
            (e.g., ``SELECT :x AS value``). Libpq-style ``%(name)s`` will also
            be translated to ``:name`` automatically.
        params : dict | list | None, optional
            Parameters to bind into the SQL. Use a dict for named parameters or
            a list/tuple for positional ones.
        conn : sqlalchemy.engine.Connection, optional
            If provided, this connection will be used; otherwise a pooled
            connection is checked out automatically.

        Returns
        -------
        pandas.DataFrame
            The query result as a DataFrame (possibly empty).

        Examples
        --------
        Basic query:

        >>> db = DB()
        >>> df = db.query_dataframe("SELECT 1 AS value")
        >>> int(df.loc[0, "value"]) == 1
        True

        With parameters:

        >>> df = db.query_dataframe("SELECT :x + :y AS s", params={"x": 2, "y": 3})
        >>> int(df.loc[0, "s"]) == 5
        True

        Reusing a single connection for multiple queries:

        >>> with db.connection() as c:
        ...     a = db.query_dataframe("SELECT 1 AS a", conn=c)
        ...     b = db.query_dataframe("SELECT 2 AS b", conn=c)
        ...     int(a.loc[0, "a"]) + int(b.loc[0, "b"]) == 3
        True
        """
        if conn is not None:
            return pd.read_sql(text(sql), params=params, con=conn)
        with self.connection() as c:
            return pd.read_sql(text(sql), params=params, con=c)

    def query_series(self, sql: str, /, *, params: dict | list | None = None,
                     conn: Connection | None = None
                     ) -> pd.Series | None:
        """Return a single row as a pandas Series, or ``None`` if no rows.

        Parameters
        ----------
        sql : str
            SQL expected to return at most one row.
        params : dict | list | None, optional
            Parameters to bind.
        conn : sqlalchemy.engine.Connection, optional
            Optional connection to reuse.

        Returns
        -------
        pandas.Series | None
            The single row as a Series if present; ``None`` if the result is
            empty.

        Raises
        ------
        ValueError
            If more than one row is returned.

        Examples
        --------
        Row present:

        >>> db = DB()
        >>> s = db.query_series("SELECT 42 AS answer")
        >>> int(s["answer"]) == 42
        True

        No rows yields None:

        >>> none = db.query_series("SELECT 1 WHERE 1=0")
        >>> none is None
        True
        """
        df = self.query_dataframe(sql, params=params, conn=conn)
        if len(df) == 0:
            return None
        if len(df) > 1:
            raise ValueError("Expected a single row for series; got multiple")
        return df.iloc[0]

    def query_rows(self, sql: str, /, *, params: dict | list | None = None,
                   conn: Connection | None = None
                   ) -> list[Any]:
        """Return all rows as a list of DBAPI row objects.

        This is a lightweight wrapper around `Connection.execute(...).fetchall()`.

        Parameters
        ----------
        sql : str
            SQL to execute.
        params : dict | list | None, optional
            Parameters to bind.
        conn : sqlalchemy.engine.Connection, optional
            Optional connection to reuse.

        Returns
        -------
        list
            List of row objects (possibly empty). Index columns by position or
            by name depending on the DBAPI driver's row type.

        Examples
        --------
        >>> db = DB()
        >>> rows = db.query_rows("SELECT 1 AS a UNION ALL SELECT 2 AS a ORDER BY a")
        >>> [int(r[0]) for r in rows]
        [1, 2]
        """
        result = self._execute(sql, params=params, conn=conn)
        return result.fetchall()

    def query_row(self, sql: str, /, *, params: dict | list | None = None,
                  conn: Connection | None = None
                  ) -> Any | None:
        """Return the first row or ``None`` if there are no rows.

        Parameters
        ----------
        sql : str
            SQL to execute.
        params : dict | list | None, optional
            Parameters to bind.
        conn : sqlalchemy.engine.Connection, optional
            Optional connection to reuse.

        Returns
        -------
        Any | None
            The first row object or ``None`` if the result set is empty.

        Examples
        --------
        >>> db = DB()
        >>> row = db.query_row("SELECT 1 AS a")
        >>> int(row[0]) == 1
        True
        """
        result = self._execute(sql, params=params, conn=conn)
        return result.fetchone()

    def query_array(self, sql: str, /, *, params: dict | list | None = None,
                    conn: Connection | None = None
                    ) -> np.ndarray:
        """Return all rows as a NumPy array of dtype=object.

        Parameters
        ----------
        sql : str
            SQL to execute.
        params : dict | list | None, optional
            Parameters to bind.
        conn : sqlalchemy.engine.Connection, optional
            Optional connection to reuse.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(n_rows, n_columns)`` and ``dtype=object``.

        Examples
        --------
        >>> db = DB()
        >>> arr = db.query_array("SELECT 1 AS a UNION ALL SELECT 2 AS a ORDER BY a")
        >>> arr.shape[0] == 2 and int(arr[0][0]) == 1 and int(arr[1][0]) == 2
        True
        """
        result = self._execute(sql, params=params, conn=conn)
        rows = result.fetchall()
        return np.array(rows, dtype=object)

    def query_scalar(self, sql: str, /, *, params: dict | list | None = None,
                     conn: Connection | None = None
                     ) -> Any | None:
        """Return the first column of the first row, or ``None`` if no rows.

        Use this when the SQL returns a single value, e.g., ``COUNT(*)``.

        Parameters
        ----------
        sql : str
            SQL to execute.
        params : dict | list | None, optional
            Parameters to bind.
        conn : sqlalchemy.engine.Connection, optional
            Optional connection to reuse.

        Returns
        -------
        Any | None
            The scalar value or ``None`` if the result set is empty.

        Examples
        --------
        >>> db = DB()
        >>> val = db.query_scalar("SELECT 2 + 3")
        >>> int(val) == 5
        True
        """
        result = self._execute(sql, params=params, conn=conn)
        row = result.fetchone()
        return None if row is None else row[0]

    def query_scalars(self, sql: str, /, *, params: dict | list | None = None,
                      conn: Connection | None = None
                      ) -> list[Any]:
        """Return the first column of all rows as a list.

        Parameters
        ----------
        sql : str
            SQL to execute.
        params : dict | list | None, optional
            Parameters to bind.
        conn : sqlalchemy.engine.Connection, optional
            Optional connection to reuse.

        Returns
        -------
        list
            A list of scalar values (possibly empty).

        Examples
        --------
        >>> db = DB()
        >>> vals = db.query_scalars("SELECT 1 UNION ALL SELECT 2 ORDER BY 1")
        >>> [int(v) for v in vals]
        [1, 2]
        """
        result = self._execute(sql, params=params, conn=conn)
        return [r[0] for r in result.fetchall()]

    def insert(self, *args, **kwargs) -> int | None:
        """Thin wrapper around `insert_dataframe`.

        Notes
        -----
        This method simply forwards all arguments to `insert_dataframe` and
        returns its result. It exists for backward compatibility with code
        that previously used `insert` for bulk inserts; it now returns the
        number of inserted rows (or None).

        Parameters
        ----------
        *args, **kwargs
            Passed directly to `insert_dataframe`.

        Returns
        -------
        int | None
            Number of inserted rows for DataFrame mode; None if no rows were inserted.

        Examples
        --------
        Basic usage returning a DataFrame:

        >>> db = DB()
        >>> df = pd.DataFrame({"id": [1, 2, 3], "name": ["alice", "bob", "charlie"]})
        >>> n = db.insert("users", df=df)
        >>> print(f"Inserted {n} rows")

        Insert with custom chunk size and include the index:

        >>> df = pd.DataFrame({"value": [10, 20, 30]}, index=["a", "b", "c"])
        >>> n = db.insert("data", df=df, index=True, chunksize=1000)
        """
        # Back-compat thin wrapper should return inserted row count
        return self.insert_dataframe(*args, **kwargs)

    def insert_dataframe(
        self,
        table: str,
        df: pd.DataFrame,
        index: bool = False,
        chunksize: int = 10000,
        **kwargs: Any,
    ) -> int | None:
        """Insert into a table via a dataframe.

        Parameters
        ----------
        table : str
            Destination table name.
        df : pandas.DataFrame
            DataFrame containing data to insert.
        index : bool, default False
            Write DataFrame index as a column when using df mode.
        chunksize : int, default 10000
            Number of rows per batch when using df mode. You shouldn't need to
            change this unless you are inserting 10_000+ rows and run into memory issues.

        Returns
        -------
        int | None
            Number of inserted rows for DataFrame mode; None if dataframe is empty.

        Raises
        ------
        TypeError
            If df is provided but is not a pandas DataFrame.

        Examples
        --------
        Insert a DataFrame with default parameters:

        >>> db = DB()
        >>> df = pd.DataFrame({"id": [1, 2, 3], "name": ["alice", "bob", "charlie"]})
        >>> n = db.insert_dataframe("users", df)
        >>> print(f"Inserted {n} rows")

        Insert with custom chunk size and include the index:

        >>> df = pd.DataFrame({"value": [10, 20, 30]}, index=["a", "b", "c"])
        >>> n = db.insert_dataframe("data", df, index=True, chunksize=1000)
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input `df` must be a pandas DataFrame.")
        if df.empty:
            self.logger.warning("Input DataFrame is empty. No data inserted.")
            return None

        try:
            self.logger.info(f"Starting insert of {len(df)} rows into table '{table}'...")
            with self.connection() as conn:
                inserted_rows = df.to_sql(
                    name=table,
                    con=conn,
                    if_exists="append",
                    index=index,
                    chunksize=chunksize,
                    method="multi",
                )

            self.logger.info(f"Successfully inserted {inserted_rows} rows into '{table}'")
            return inserted_rows
        except Exception as e:
            self.logger.error(f"Failed to insert data into '{table}' using to_sql: {e}")
            raise

    def insert_kw(self, table: str, **kwargs: Any) -> None:
        """ Insert a single row using keyword arguments.

        This method provides an explicit keyword-argument interface for inserting
        a single row into a table. Column names are provided as keyword arguments
        with their corresponding values.

        Parameters
        ----------
        table : str
            The name of the destination table.
        **kwargs : Any
            Column-value pairs for the row to insert. Keys must match column names
            in the target table.

        Returns
        -------
        None

        Examples
        --------
        Insert a single row with explicit column values:

        >>> db = DB()
        >>> db.insert_kw("pfs_visit", pfs_visit_id=1, description="first visit")
        """

        # Single-row insert path (legacy behavior)
        md = MetaData()
        t = Table(table, md, autoload_with=self.engine)
        ins = t.insert().values(**kwargs)
        with self.connection() as conn:
            conn.execute(ins)

        return None
