import sqlite3
import polars as pl
import numpy as np
import json
import io

from more_itertools import one, flatten, unique_everseen as unique
from urllib.request import pathname2url
from pathlib import Path
from textwrap import dedent

from typing import Any, Literal

# Note: functions that accept a `zone_id` and use it to make a query should 
# always explicitly convert the ID to an integer.  If `numpy` was involved in 
# generating the ID in question, it might actually be a `np.int64` rather than 
# a plain integer.  This will be interpreted by SQLite as a blob, and as a 
# result the query will mysteriously fail.

def hash_db(db_path: str | Path):
    import xxhash
    m = xxhash.xxh3_64()
    with open(db_path, 'rb') as f:
        while chunk := f.read(2**20):
            m.update(chunk)
    return m.hexdigest()

def open_db(path: str | Path, mode: Literal['ro', 'rw', 'rwc'] = 'ro'):
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

    sqlite3.register_adapter(list, _adapt_json)
    sqlite3.register_converter('LIST', _convert_json)

    sqlite3.register_adapter(np.ndarray, _adapt_array)
    sqlite3.register_converter('VECTOR_3D', _convert_array)

    sqlite3.register_adapter(pl.DataFrame, _adapt_dataframe)
    sqlite3.register_converter('ATOMS', _convert_dataframe)

    if isinstance(path, Path):
        path = pathname2url(str(path))

    # https://stackoverflow.com/questions/4699605/why-doesn-t-sqlite-require-a-commit-call-to-save-data
    db = sqlite3.connect(
            f'file:{path}?mode={mode}',
            uri=True,
            isolation_level='DEFERRED',
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    )
    db.execute('PRAGMA foreign_keys = ON')

    return db

def init_db(db):
    cur = db.cursor()

    cur.execute('''\
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value ANY
            )
    ''')
    cur.execute('''\
            CREATE TABLE IF NOT EXISTS split (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            )
    ''')
    cur.execute('''\
            CREATE TABLE IF NOT EXISTS structure (
                id INTEGER PRIMARY KEY,
                split_id INTEGER REFERENCES split(id),
                pdb_id TEXT UNIQUE NOT NULL,
                model_id TEXT NOT NULL
            )
    ''')
    cur.execute('''\
            CREATE TABLE IF NOT EXISTS assembly (
                id INTEGER PRIMARY KEY,
                struct_id INTEGER NOT NULL REFERENCES structure(id),
                pdb_id TEXT NOT NULL,
                atoms ATOMS NOT NULL,
                UNIQUE (struct_id, pdb_id)
            )
    ''')
    cur.execute('''\
            CREATE TABLE IF NOT EXISTS zone (
                id INTEGER PRIMARY KEY,
                assembly_id INTEGER NOT NULL REFERENCES assembly(id),
                center_A VECTOR_3D NOT NULL
            )
    ''')
    cur.execute('''\
            CREATE TABLE IF NOT EXISTS subchain (
                zone_id INTEGER NOT NULL REFERENCES zone(id),
                pdb_id TEXT NOT NULL,
                symmetry_mate INTEGER NOT NULL
            )
    ''')
    cur.execute('''\
            CREATE TABLE IF NOT EXISTS subchain_pair (
                zone_id INTEGER NOT NULL REFERENCES zone(id),
                pdb_id_1 TEXT NOT NULL,
                symmetry_mate_1 INTEGER NOT NULL,
                pdb_id_2 TEXT NOT NULL,
                symmetry_mate_2 INTEGER NOT NULL
            )
    ''')
    cur.execute('''\
            CREATE TABLE IF NOT EXISTS neighbor (
                id INTEGER PRIMARY KEY,
                offset_A VECTOR_3D
            )
    ''')
    cur.execute('''\
            CREATE TABLE IF NOT EXISTS zone_neighbor (
                zone_id INTEGER NOT NULL REFERENCES zone(id),
                neighbor_id INTEGER NOT NULL REFERENCES neighbor(id)
            )
    ''')
    cur.execute('''\
            CREATE TABLE IF NOT EXISTS curriculum (
                zone_id INTEGER NOT NULL REFERENCES zone(id),
                random_seed INTEGER NOT NULL,
                difficulty REAL NOT NULL,
                UNIQUE (zone_id, random_seed)
            )
    ''')


def upsert_metadata(db, metadata: dict[str, Any]):
    db.executemany(
            '''\
            INSERT INTO metadata (key, value)
            VALUES (:k, :v)
            ON CONFLICT (key)
            DO UPDATE SET value=:v
            ''',
            [dict(k=k, v=v) for k, v in metadata.items()],
    )

def insert_structure(db, pdb_id, *, model_id):
    cur = db.execute(
            'SELECT id FROM structure WHERE pdb_id=?',
            [pdb_id],
    )
    cur.row_factory = _scalar_row_factory

    ids = cur.fetchall()
    if ids:
        return one(ids)

    cur = db.execute(
            '''\
            INSERT INTO structure (pdb_id, model_id)
            VALUES (?, ?)
            RETURNING id
            ''',
            [pdb_id, model_id],
    )
    cur.row_factory = _scalar_row_factory
    return one(cur.fetchall())

def update_splits(db, splits):
    assert set(splits.keys()) == set(select_structures(db))

    db.execute('DELETE FROM split')
    db.executemany(
            'INSERT INTO split (name) VALUES (?)',
            [(x,) for x in unique(splits.values())],
    )
    cur = db.execute('SELECT name, id FROM split')
    split_ids = dict(cur.fetchall())

    db.executemany(
            'UPDATE structure SET split_id=? WHERE pdb_id=?',
            [(split_ids[split], pdb_id) for pdb_id, split in splits.items()]
    )

def insert_assembly(db, struct_id, pdb_id, atoms):
    cur = db.execute(
            '''\
            INSERT INTO assembly (struct_id, pdb_id, atoms)
            VALUES (?, ?, ?)
            RETURNING id
            ''',
            [struct_id, pdb_id, atoms],
    )
    cur.row_factory = _scalar_row_factory
    return one(cur.fetchall())

def insert_zone(
        db,
        assembly_id,
        *,
        center_A,
        neighbor_ids,
        subchains=[],
        subchain_pairs=[],
):
    cur = db.execute(
            '''\
            INSERT INTO zone (assembly_id, center_A)
            VALUES (?, ?)
            RETURNING id
            ''',
            [assembly_id, center_A],
    )
    zone_id = one(one(cur.fetchall()))

    db.executemany(
            '''\
            INSERT INTO subchain (zone_id, pdb_id, symmetry_mate)
            VALUES (?, ?, ?)
            ''',
            [(zone_id, *subchain) for subchain in subchains]
    )
    db.executemany(
            '''\
            INSERT INTO subchain_pair (
                zone_id,
                pdb_id_1, symmetry_mate_1,
                pdb_id_2, symmetry_mate_2
            )
            VALUES (?, ?, ?, ?, ?)
            ''',
            [(zone_id, *flatten(sorted(pair))) for pair in subchain_pairs],
    )
    db.executemany(
            '''\
            INSERT INTO zone_neighbor (zone_id, neighbor_id)
            VALUES (?, ?)
            ''',
            [(zone_id, neighbor_id) for neighbor_id in neighbor_ids],
    )
    return zone_id

def insert_neighbors(db, offsets_A):
    assert select_neighbors(db).size == 0
    db.executemany(
            'INSERT INTO neighbor (id, offset_A) VALUES (?, ?)',
            list(enumerate(offsets_A)),
    )

def insert_curriculum(db, zone_ids, random_seeds, difficulty):
    params = zip(zone_ids, random_seeds, difficulty, strict=True)
    db.executemany('''\
            INSERT INTO curriculum (zone_id, random_seed, difficulty)
            VALUES (?, ?, ?)
    ''', params)


def select_metadata(db, keys):
    placeholders = ', '.join(['?'] * len(keys))
    cur = db.execute(
            f'SELECT key, value FROM metadata WHERE key in ({placeholders})',
            keys,
    )
    return dict(cur.fetchall())

def select_metadatum(db, key):
    cur = db.execute('SELECT value FROM metadata WHERE key=?', [key])
    cur.row_factory = _scalar_row_factory
    return one(
            cur.fetchall(),
            too_short=KeyError(key),
    )

def select_structures(db):
    cur = db.execute('SELECT pdb_id FROM structure')
    cur.row_factory = _scalar_row_factory
    return cur.fetchall()

def select_split(db, split):
    cur = db.execute('''\
            SELECT zone.id
            FROM zone
            JOIN assembly ON assembly.id = zone.assembly_id
            JOIN structure ON structure.id = assembly.struct_id
            JOIN split ON split.id = structure.split_id
            WHERE split.name = ?
            ORDER BY zone.id
    ''', [split])
    cur.row_factory = _scalar_row_factory
    return np.array(cur.fetchall())

def select_zone_ids(db):
    """
    Return a list of every zone id in the database.

    .. warning::
        
        Don't use this function during training.  For one thing, it doesn't 
        distinguish between the different splits (e.g. train/test/validation).  
        For another, it isn't guaranteed to return the zone in any particular 
        order, which could lead to non-deterministic behavior.
    """
    cur = db.execute('SELECT zone.id FROM zone')
    cur.row_factory = _scalar_row_factory
    return cur.fetchall()

def select_zone_pdb_ids(db, zone_id):
    cur = db.execute('''\
            SELECT 
                structure.pdb_id AS struct_pdb_id,
                structure.model_id AS model_pdb_id,
                assembly.pdb_id AS assembly_pdb_id
            FROM structure
            JOIN assembly ON structure.id = assembly.struct_id
            JOIN zone ON assembly.id = zone.assembly_id
            WHERE zone.id=?
        ''', [int(zone_id)])
    cur.row_factory = _dict_row_factory
    return one(cur.fetchall())

def select_zone_center_A(db, zone_id):
    cur = db.execute(
            'SELECT center_A FROM zone WHERE id=?',
            [int(zone_id)],
    )
    cur.row_factory = _scalar_row_factory
    return one(cur.fetchall())

def select_zone_atoms(db, zone_id):
    cur = db.execute('''\
            SELECT assembly.atoms
            FROM assembly
            JOIN zone ON assembly.id = zone.assembly_id
            WHERE zone.id=?
    ''', [int(zone_id)])
    cur.row_factory = _scalar_row_factory
    return one(cur.fetchall())

def select_zone_subchains(db, zone_id):
    cur = db.execute('''\
            SELECT pdb_id, symmetry_mate
            FROM subchain
            WHERE zone_id = ?
    ''', [int(zone_id)])
    subchains = cur.fetchall()

    cur = db.execute('''\
            SELECT pdb_id_1, symmetry_mate_1, pdb_id_2, symmetry_mate_2
            FROM subchain_pair
            WHERE zone_id = ?
    ''', [zone_id])
    subchain_pairs = [((a, b), (c, d)) for a, b, c, d in cur.fetchall()]

    return subchains, subchain_pairs

def select_zone_neighbors(db, zone_id):
    cur = db.execute('''\
            SELECT neighbor_id
            FROM zone_neighbor
            WHERE zone_id=?
    ''', [int(zone_id)])
    cur.row_factory = _scalar_row_factory
    return cur.fetchall()

def select_neighbors(db):
    rows = db.execute('SELECT id, offset_A FROM neighbor').fetchall()
    return np.array([
        offset_A
        for _, offset_A in sorted(rows)
    ])

def select_curriculum(db, max_difficulty):
    cur = db.execute('''\
            SELECT zone_id
            FROM (
                SELECT zone_id, AVG(difficulty) AS difficulty
                FROM curriculum
                GROUP BY zone_id
            )
            WHERE difficulty < ?
    ''', [max_difficulty])
    cur.row_factory = _scalar_row_factory
    return np.array(cur.fetchall())

def select_max_curriculum_seed(db, default=0):
    cur = db.execute('SELECT MAX(random_seed) FROM curriculum')
    cur.row_factory = _scalar_row_factory
    return cur.fetchone() or default
    
def select_dataframe(db, query):
    cur = db.execute(query)
    cur.row_factory = _dict_row_factory
    return pl.DataFrame(cur.fetchall())


def delete_metadatum(db, key):
    db.execute('DELETE FROM metadata WHERE key=?', [key])

def delete_splits(db):
    db.execute('UPDATE structure SET split_id=NULL')
    db.execute('DELETE FROM split')


def show(db, query):
    df = select_dataframe(db, query)
    print(dedent(query))
    print(df)


def _adapt_json(obj):
    return json.dumps(obj).encode('utf-8')

def _convert_json(bytes):
    return json.loads(bytes.decode('utf-8'))

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

