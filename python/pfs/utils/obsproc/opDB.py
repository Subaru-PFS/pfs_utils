#!/usr/bin/env python

from sqlalchemy import create_engine


def get_url(conf):
    url = f'{conf["dialect"]}://{conf["user"]}@{conf["host"]}:{conf["port"]}/{conf["dbname"]}'
    return url 


class OpDB(object):
    """OpDB

    Parameters
    ----------
    dbConfig : DB config read from toml file


    Examples
    ----------

    """
    def __init__(self, conf):
        self.conf = conf
        self._engine = create_engine(get_url(self.conf))
        self._conn = self._engine.raw_connection()
