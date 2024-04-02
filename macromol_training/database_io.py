import sqlite3
import polars as pl
import numpy as np
import io

from more_itertools import one

def open_db(path):
    """
    .. warning::
        It's not safe to fork the database connection object returned by this 
        function.  Thus, either avoid using the ``"fork"`` multiprocessing 
        context (e.g. with ``torch.utils.data.DataLoader``), or don't open the 
        database until already within the subprocess.

    .. warning::
        The database connection returned by this function does not have 
        autocommit behavior enabled, so the caller is responsible for 
        committing/rolling back transactions as necessary.
    """

    sqlite3.register_adapter(np.ndarray, _adapt_array)
    sqlite3.register_converter('3D_VECTOR', _convert_array)

    db = sqlite3.connect(
            path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    )
    db.execute('PRAGMA foreign_keys = ON')

    return db

def init_splits(db):
    cur = db.cursor()

    cur.execute('''\
            CREATE TABLE IF NOT EXISTS meta (
                key UNIQUE,
                value
            )
    ''')
    cur.execute('''\
            CREATE TABLE IF NOT EXISTS origins (
                id INTEGER PRIMARY KEY,
                pdb_id TEXT UNIQUE,
                center_A 3D_VECTOR,
                split TEXT,
            )
    ''')
    cur.execute('''\
            CREATE TABLE IF NOT EXISTS neighbors (
                id INTEGER PRIMARY KEY,
                offset_A 3D_VECTOR,
            )
    ''')
    cur.execute('''\
            CREATE TABLE IF NOT EXISTS origin_neighbors (
                origin_id,
                neighbor_id,
            )
    ''')


def _adapt_array(array):
    out = io.BytesIO()
    np.save(out, array, allow_pickle=False)
    return out.getvalue()

def _convert_array(bytes):
    in_ = io.BytesIO(bytes)
    x = np.load(in_)
    return x

def _adapt_dataframe(df):
    out = io.BytesIO()
    df.write_parquet(out)
    return out.getvalue()

def _convert_dataframe(bytes):
    in_ = io.BytesIO(bytes)
    df = pl.read_parquet(in_)
    return df

def _dict_row_factory(cur, row):
    return {
            desc[0]: v
            for desc, v in zip(cur.description, row)
    }

def _dataclass_row_factory(cls, col_map={}):

    def factory(cur, row):
        row_dict = {
                col_map.get(k := col[0], k): value
                for col, value in zip(cur.description, row)
        }
        return cls(**row_dict)

    return factory

def _scalar_row_factory(cur, row):
    return one(row)

