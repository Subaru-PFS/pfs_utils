import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from pfs.utils.database.db import DB
from pfs.utils.database.gaia import GaiaDB
from pfs.utils.database.opdb import OpDB


@pytest.fixture()
def sqlite_db(tmp_path):
    """Provide a DB instance backed by a temporary SQLite file.

    We monkeypatch DB.url to use sqlite so we can exercise the logic
    without requiring a Postgres server/driver.
    """

    # Create a temporary sqlite database file
    dbfile = tmp_path / "test.sqlite"

    class SqliteDB(DB):
        @property
        def url(self) -> str:  # type: ignore[override]
            return f"sqlite+pysqlite:///{dbfile}"

    db = SqliteDB(dbname="testdb", user="user", host="localhost")

    # Create a sample table to use in tests
    db.commit(
        """
        CREATE TABLE IF NOT EXISTS sample
        (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )
        """
    )

    # Ensure table starts empty
    db.commit("DELETE FROM sample")

    return db


def test_dsn_default_and_setter_updates_attributes():
    db = DB(dbname="opdb", user="pfs", host="db-ics", port=15432)
    # Default dsn constructed from attributes
    assert "dbname=opdb" in db.dsn
    assert "user=pfs" in db.dsn
    assert "host=db-ics" in db.dsn
    assert "port=15432" in db.dsn

    # Setting DSN via string updates attributes
    db.dsn = "dbname=foo user=bar host=baz port=5439"
    assert db.dbname == "foo"
    assert db.user == "bar"
    assert db.host == "baz"
    assert db.port == 5439

    # Setting DSN via mapping updates attributes
    db.dsn = {"dbname": "qq", "user": "uu", "host": "hh", "port": 6000}
    assert db.dbname == "qq"
    assert db.user == "uu"
    assert db.host == "hh"
    assert db.port == 6000


def test_build_url_is_postgres_by_default():
    db = DB(dbname="opdb", user="pfs", host="db-ics", port=5432)
    url = db.url
    assert url.startswith("postgresql+psycopg://pfs@db-ics:5432/opdb")


def test_engine_is_cached(sqlite_db):
    e1 = sqlite_db.engine
    e2 = sqlite_db.engine
    assert e1 is e2


def test_connect_and_connection_context(sqlite_db):
    # Insert using the main API path
    sqlite_db.insert_dataframe("sample", df=pd.DataFrame({"id": [1], "name": ["a"]}))
    name = sqlite_db.query_scalar(
        "SELECT name FROM sample WHERE id = :id",
        params={"id": 1},
    )
    assert name == "a"

    sqlite_db.insert_kw("sample", id=2, name="b")

    name = sqlite_db.query_scalar(
        "SELECT name FROM sample WHERE id = :id",
        params={"id": 2},
    )
    assert name == "b"


def test_fetch_dataframe_and_single_as_series(sqlite_db):
    # Add two rows via insert
    sqlite_db.insert_dataframe("sample", df=pd.DataFrame({"id": [1, 2], "name": ["x", "y"]}))

    df = sqlite_db.query_dataframe("SELECT * FROM sample ORDER BY id")
    assert isinstance(df, pd.DataFrame)
    assert list(df["name"]) == ["x", "y"]

    # Single row returns Series if requested
    series = sqlite_db.query_series(
        "SELECT * FROM sample WHERE id = :id",
        params={"id": 1},
    )
    assert isinstance(series, pd.Series)
    assert series["name"] == "x"


def test_query_rows_and_query_scalars(sqlite_db):
    # prepare rows
    sqlite_db.insert_dataframe("sample", df=pd.DataFrame({"id": [3, 4], "name": ["ccc", "ddd"]}))

    rows = sqlite_db.query_rows("SELECT id, name FROM sample ORDER BY id")
    assert [tuple(r) for r in rows] == [(3, "ccc"), (4, "ddd")]

    ids = sqlite_db.query_scalars("SELECT id FROM sample ORDER BY id")
    assert ids == [3, 4]


def test_query_series_empty_and_multiple(sqlite_db):
    # empty -> None
    res = sqlite_db.query_series("SELECT * FROM sample WHERE id = :id", params={"id": 9999})
    assert res is None

    # multiple -> ValueError
    sqlite_db.insert_dataframe("sample", df=pd.DataFrame({"id": [11, 12], "name": ["aa", "bb"]}))
    with pytest.raises(ValueError):
        sqlite_db.query_series("SELECT * FROM sample")


def test_execute_percent_named_params(sqlite_db):
    # ensure %(name)s style is translated to :name
    sqlite_db.insert_kw("sample", id=21, name="twentyone")
    name = sqlite_db.query_scalar("SELECT name FROM sample WHERE id = %(id)s", params={"id": 21})
    assert name == "twentyone"


def test_query_row_none_when_no_match(sqlite_db):
    res = sqlite_db.query_row("SELECT id, name FROM sample WHERE id = :id", params={"id": -1})
    assert res is None


def test_fetchall_and_fetchone_numpy_arrays(sqlite_db):
    sqlite_db.insert_dataframe("sample", df=pd.DataFrame({"id": [10, 20], "name": ["ten", "twenty"]}))

    arr = sqlite_db.query_array("SELECT id, name FROM sample ORDER BY id")
    assert isinstance(arr, np.ndarray)
    # shape (nrows,) of Row objects; compare by tuple
    assert [tuple(r) for r in arr.tolist()] == [(10, "ten"), (20, "twenty")]

    one = sqlite_db.query_row(
        "SELECT id, name FROM sample WHERE id = :id",
        params={"id": 20},
    )
    assert one is not None
    assert tuple(one) == (20, "twenty")


def test_commit_executes_and_persists(sqlite_db):
    sqlite_db.insert_kw("sample", id=7, name="seven")

    n = sqlite_db.query_scalar(
        "SELECT COUNT(*) AS n FROM sample WHERE id = :id",
        params={"id": 7},
    )
    assert int(n) == 1


def test_insert_dataframe_inserts_rows(sqlite_db):
    # Create table that matches DataFrame columns
    sqlite_db.commit(
        """
        CREATE TABLE IF NOT EXISTS df_table
        (
            a INTEGER,
            b TEXT
        )
        """
    )

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    n = sqlite_db.insert_dataframe("df_table", df=df)
    assert n == 3

    out = sqlite_db.query_dataframe("SELECT a, b FROM df_table ORDER BY a")
    assert list(out["a"]) == [1, 2, 3]
    assert list(out["b"]) == ["x", "y", "z"]


def test_insert_dataframe_empty_returns_none(sqlite_db):
    # Pre-create table
    sqlite_db.commit("CREATE TABLE IF NOT EXISTS empty_table (x INTEGER)")
    empty = pd.DataFrame(columns=["x"]).astype({"x": "int64"})
    res = sqlite_db.insert_dataframe("empty_table", df=empty)
    assert res is None


def test_query_wrapper_returns_dataframe(sqlite_db):
    # use wrapper `query` (back-compat) and ensure it returns a DataFrame
    sqlite_db.insert_kw("sample", id=101, name="wrapped")
    df = sqlite_db.query("SELECT id, name FROM sample WHERE id = :id", params={"id": 101})
    assert isinstance(df, pd.DataFrame)
    assert int(df.loc[0, "id"]) == 101
    assert df.loc[0, "name"] == "wrapped"


def test_insert_wrapper_returns_count(sqlite_db):
    # Create a separate table for this test
    sqlite_db.commit("CREATE TABLE IF NOT EXISTS wrap_table (a INTEGER, b TEXT)")
    df = pd.DataFrame({"a": [1, 2], "b": ["u", "v"]})
    n = sqlite_db.insert("wrap_table", df)
    assert n == 2
    out = sqlite_db.query_dataframe("SELECT a, b FROM wrap_table ORDER BY a")
    assert list(out["a"]) == [1, 2]
    assert list(out["b"]) == ["u", "v"]


def test_connect_returns_connection_and_executes(sqlite_db):
    # Explicitly use connect() and run a trivial statement
    conn = sqlite_db.connect()
    try:
        val = conn.execute(text("SELECT 1")).scalar()
        assert int(val) == 1
    finally:
        conn.close()


def test_connection_context_allows_multiple_executes(sqlite_db):
    # Explicit connection() context for multiple operations
    with sqlite_db.connection() as conn:
        conn.execute(text("INSERT INTO sample(id, name) VALUES(:i, :n)"), {"i": 301, "n": "aaa"})
        conn.execute(text("INSERT INTO sample(id, name) VALUES(:i, :n)"), {"i": 302, "n": "bbb"})
    names = sqlite_db.query_scalars("SELECT name FROM sample WHERE id IN (301,302) ORDER BY id")
    assert names == ["aaa", "bbb"]


def test_gaia_commit_raises_not_implemented():
    gdb = GaiaDB()
    with pytest.raises(NotImplementedError):
        gdb.commit("SELECT 1")


def test_gaia_insert_dataframe_raises_not_implemented():
    gdb = GaiaDB()
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(NotImplementedError):
        gdb.insert_dataframe("some_table", df=df)


def test_gaia_insert_kw_raises_not_implemented():
    gdb = GaiaDB()
    with pytest.raises(NotImplementedError):
        gdb.insert_kw("some_table", a=1)


# -----------------------------------------------------------------------------
# Tests for class-scoped default connection parameters and subclass overrides
# -----------------------------------------------------------------------------


@pytest.fixture()
def restore_defaults():
    """Snapshot and restore global and class-level defaults after a test.

    This prevents cross-test contamination when exercising
    set_default_connection on DB and subclasses.
    """
    # Snapshot module-wide defaults from DB module
    from pfs.utils.database import db as db_module

    snap = {
        "DEFAULT_HOST": db_module.DEFAULT_HOST,
        "DEFAULT_USER": db_module.DEFAULT_USER,
        "DEFAULT_DBNAME": db_module.DEFAULT_DBNAME,
        "DEFAULT_PORT": db_module.DEFAULT_PORT,
        # Snapshot subclass class attributes
        "OpDB": {
            "host": getattr(OpDB, "host", None),
            "user": getattr(OpDB, "user", None),
            "dbname": getattr(OpDB, "dbname", None),
            "port": getattr(OpDB, "port", None),
        },
        "GaiaDB": {
            "host": getattr(GaiaDB, "host", None),
            "user": getattr(GaiaDB, "user", None),
            "dbname": getattr(GaiaDB, "dbname", None),
            "port": getattr(GaiaDB, "port", None),
        },
    }

    yield

    # Restore DB module globals
    db_module.DEFAULT_HOST = snap["DEFAULT_HOST"]
    db_module.DEFAULT_USER = snap["DEFAULT_USER"]
    db_module.DEFAULT_DBNAME = snap["DEFAULT_DBNAME"]
    db_module.DEFAULT_PORT = snap["DEFAULT_PORT"]

    # Restore subclass class attributes
    for cls, saved in ((OpDB, snap["OpDB"]), (GaiaDB, snap["GaiaDB"])):
        setattr(cls, "host", saved["host"])
        setattr(cls, "user", saved["user"])
        setattr(cls, "dbname", saved["dbname"])
        setattr(cls, "port", saved["port"])


def test_global_defaults_affect_DB_only(restore_defaults):
    # Change global defaults via DB
    DB.set_default_connection(host="glob-host", user="glob-user", dbname="glob-db", port=15432)

    # DB instances pick up global defaults
    db = DB()
    assert (db.host, db.user, db.dbname, db.port) == ("glob-host", "glob-user", "glob-db", 15432)

    # Subclasses with their own class-level defaults should remain unchanged
    op = OpDB()
    assert (op.host, op.user, op.dbname, op.port) == ("db-ics", "pfs", "opdb", 5432)

    gaia = GaiaDB()
    assert (gaia.host, gaia.user, gaia.dbname, gaia.port) == ("g2sim-cat", "obsuser", "star_catalog", 5438)


def test_subclass_defaults_only_affect_that_subclass(restore_defaults):
    # Change OpDB defaults only
    OpDB.set_default_connection(host="opdb-host", user="opdb-user", dbname="opdb-name", port=5555)

    # OpDB instances reflect the new class defaults
    op1 = OpDB()
    assert (op1.host, op1.user, op1.dbname, op1.port) == ("opdb-host", "opdb-user", "opdb-name", 5555)

    # DB and GaiaDB unaffected
    db = DB()
    # DB uses module-wide defaults, which are the library defaults unless changed
    assert (db.host, db.user, db.dbname, db.port) == ("localhost", "user", "dbname", 5432)

    gaia = GaiaDB()
    assert (gaia.host, gaia.user, gaia.dbname, gaia.port) == ("g2sim-cat", "obsuser", "star_catalog", 5438)


def test_init_args_take_precedence_over_class_defaults(restore_defaults):
    # Set subclass defaults
    OpDB.set_default_connection(host="cls-host", user="cls-user", dbname="cls-db", port=6000)

    # Explicit args should override the class defaults
    op = OpDB(host="arg-host", user="arg-user", dbname="arg-db", port=7000)
    assert (op.host, op.user, op.dbname, op.port) == ("arg-host", "arg-user", "arg-db", 7000)


def test_existing_instances_unaffected_by_later_changes(restore_defaults):
    # Create instance with current OpDB defaults
    op_before = OpDB()

    # Change defaults after instance creation
    OpDB.set_default_connection(host="new-host", user="new-user", dbname="new-db", port=5001)

    # Existing instance should retain its original attributes
    assert (op_before.host, op_before.user, op_before.dbname, op_before.port) != (
        "new-host",
        "new-user",
        "new-db",
        5001,
    )

    # New instance reflects the updated defaults
    op_after = OpDB()
    assert (op_after.host, op_after.user, op_after.dbname, op_after.port) == (
        "new-host",
        "new-user",
        "new-db",
        5001,
    )
