import numpy as np
import psycopg2


class opDB:
    """placeholder to retrieve/insert data from opDB"""
    host = 'db-ics'

    @staticmethod
    def connect():
        """ return connection object, password needs to be defined in /home/user/.pgpass """
        return psycopg2.connect(dbname='opdb', user='pfs', host=opDB.host)

    @staticmethod
    def fetchall(query):
        """ fetch all rows from query """
        with opDB.connect() as conn:
            with conn.cursor() as curs:
                curs.execute(query)
                return np.array(curs.fetchall())

    @staticmethod
    def fetchone(query):
        """ fetch one row from query """
        with opDB.connect() as conn:
            with conn.cursor() as curs:
                curs.execute(query)
                return np.array(curs.fetchone())

    @staticmethod
    def commit(query, kwargs):
        """ execute query and commit """
        with opDB.connect() as conn:
            with conn.cursor() as curs:
                curs.execute(query, kwargs)
            conn.commit()

    @staticmethod
    def insert(table, **kwargs):
        """ insert row in table, column names and values are parsed as kwargs
        Args:
            table (str): table name
        Examples:
            >>> opDB.insert('pfs_visit', pfs_visit_id=1, pfs_visit_description='i am the first pfs visit')
        """
        fields = ', '.join(kwargs.keys())
        values = ', '.join(['%%(%s)s' % v for v in kwargs])
        query = f'INSERT INTO {table} ({fields}) VALUES ({values})'
        return opDB.commit(query, kwargs)
