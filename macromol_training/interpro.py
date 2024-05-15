"""
Cluster PDB ids based on the InterPro domains and families they contain.

Usage:
    mmt_fetch_interpro_clusters <db> [-o <csv>] [-c <sqlite>] [-r <int>]

Options:
    -o --output <csv>           [default: interpro_clusters.csv]
        The path to the CSV file where the clusters should be written.  This 
        file will have two columns.  The first contains PDB ids.  The second 
        contains InterPro entry ids, each of which refers to a "domain" or 
        "family" as defined by InterPro.  Any two PDB ids that share a InterPro 
        domain or family should be considered to cluster together.
        
    -c --cache <sqlite>         [default: interpro_cache.sqlite]
        The path where responses from the InterPro server should be cached.  
        This cache is useful in the case that not all of the InterPro queries 
        are successful the first time, due to random network errors.  Once the 
        final clusters have been created, this cache can be deleted.

    -r --max-requests <int>     [default: 50]
        The maximum simultaneous number of requests to make to the InterPro web 
        server at any one time.  This is an important parameter.  Too few, and 
        we waste time waiting to hear back from the server.  Too many, and we 
        can overwhelm the server, or even our own resources (i.e. open socket 
        limits).  The default of 50 should balance these concerns pretty well.  
        However, you might try a smaller number if you're getting a lot of 
        "server disconnected" errors.
"""

import aiohttp
import asyncio
import sqlite3
import polars as pl

from .database_io import (
        open_db as open_train_db, select_structures,
        _dict_row_factory, _scalar_row_factory,
)
from more_itertools import one
from tqdm import tqdm

def main():
    import docopt
    args = docopt.docopt(__doc__)

    train_db = open_train_db(args['<db>'], mode='rw')
    cache_db = open_cache_db(args['--cache'])
    pdb_ids = find_pdb_ids_to_fetch(train_db, cache_db)

    if pdb_ids:
        # I experimented with different numbers of simultaneous connections.  
        # After ≈50 connections, I observed diminishing returns in terms of 
        # speed.  That's how I chose the default value for the `--max-requests` 
        # argument:
        #
        # Connections  Downloads
        # -----------  ---------
        # 10           15/s
        # 20           25/s
        # 50           28/s
        # 100          31/s

        asyncio.run(
                cache_interpro_entries(
                    cache_db, pdb_ids,
                    max_simultaneous_requests=int(args['--max-requests']),
                )
        )

    summarize_downloads(cache_db, pdb_ids)

    df = make_clusters(cache_db)
    df.write_csv(args['--output'], include_header=False)

def find_pdb_ids_to_fetch(train_db, cache_db):
    return select_needed_pdb_ids(train_db) - select_cached_pdb_ids(cache_db)

async def cache_interpro_entries(db, pdb_ids, *, max_simultaneous_requests):
    semaphore = asyncio.Semaphore(max_simultaneous_requests)
    queue = asyncio.Queue()
    pdb_ids = set(pdb_ids)

    async def fetch(pdb_id):
        try:
            async with semaphore:
                response = await fetch_interpro_entries_in_structure(
                        session, pdb_id, 
                )
                queue.put_nowait(response)
        finally:
            pdb_ids.discard(pdb_id)

    async with aiohttp.ClientSession() as session:

        # Have to keep references to each task, so they don't get 
        # garbage-collected before they actually trigger.
        tasks = [
                asyncio.create_task(fetch(pdb_id))
                for pdb_id in pdb_ids
        ]

        progress = tqdm(total=len(pdb_ids))
        
        while pdb_ids or not queue.empty():
            progress.update()

            response = await queue.get()
            insert_interpro_response(db, response)

        del tasks

async def fetch_interpro_entries_in_structure(session, pdb_id):
    # In most cases it doesn't matter if the PDB id is upper or lower case.  
    # However, "1set" and "7set" both return errors while "1SET" and "7SET" 
    # don't.  Given this, always use upper case.
    url = f'https://www.ebi.ac.uk/interpro/api/entry/interpro/structure/pdb/{pdb_id.upper()}'

    async with session.get(url) as response:
        if response.status == 204:
            return dict(
                    pdb_id=pdb_id,
                    status='error: not found',
                    payload=[],
            )

        payload = await response.json()

        try:
            entries = [
                    dict(
                        interpro_id=entry['metadata']['accession'],
                        type=entry['metadata']['type'],
                    )
                    for entry in payload['results']
            ]
        except KeyError:
            return dict(
                    pdb_id=pdb_id.lower(),
                    status='error: malformed payload',
                    payload=[],
            )

        return dict(
                pdb_id=pdb_id,
                status='ok',
                payload=entries,
        )

def summarize_downloads(db, pdb_ids):
    cur = db.execute('SELECT pdb_id, status FROM header')
    cur.row_factory = _dict_row_factory

    df_all = pl.DataFrame(cur.fetchall())
    df_new = df_all.filter(pl.col('pdb_id').is_in(pdb_ids))

    def count_status(status):
        return df_all.filter(status=status).height

    print(f"New downloads:       {df_new.height}")
    print(f"All downloads:       {df_all.height}")
    print()
    print(f"Success:             {count_status('ok')}")
    print(f"PDB not in InterPro: {count_status('error: not found')}")
    print(f"Malformed response:  {count_status('error: malformed payload')}")

def make_clusters(db):
    return (
            select_interpro_responses(db)
            .filter(
                pl.col('interpro_type').is_in(['domain', 'family'])
            )
            .select('pdb_id', 'interpro_id')
    )


def open_cache_db(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA foreign_keys = ON')
    db.execute('''\
            CREATE TABLE IF NOT EXISTS header (
                id INTEGER PRIMARY KEY,
                pdb_id TEXT UNIQUE,
                status TEXT
            )
    ''')
    db.execute('''\
            CREATE TABLE IF NOT EXISTS payload (
                header_id INTEGER,
                interpro_id TEXT,
                interpro_type TEXT,
                FOREIGN KEY (header_id) REFERENCES header(id)
            )
    ''')
    return db

def select_needed_pdb_ids(db):
    return set(select_structures(db))

def select_cached_pdb_ids(db):
    cur = db.execute('SELECT pdb_id FROM header')
    cur.row_factory = _scalar_row_factory
    return set(cur.fetchall())

def select_interpro_responses(db):
    cur = db.execute('''\
            SELECT pdb_id, interpro_id, interpro_type
            FROM header
            INNER JOIN payload ON header.id = payload.header_id
    ''')
    cur.row_factory = _dict_row_factory
    return pl.DataFrame(cur.fetchall())

def insert_interpro_response(db, response):
    db.execute('BEGIN')

    try:
        cur = db.execute(
                'INSERT INTO header (pdb_id, status) VALUES (?, ?) RETURNING id',
                (response['pdb_id'], response['status']),
        )
        cur.row_factory = _scalar_row_factory

        header_id = one(cur.fetchall())
        payload = [
                (header_id, d['interpro_id'], d['type'])
                for d in response['payload']
        ]

        db.executemany(
                '''\
                INSERT INTO payload (header_id, interpro_id, interpro_type)
                VALUES (?, ?, ?)
                ''',
                payload,
        )

    except:
        db.execute('ROLLBACK')
        raise
    
    else:
        db.execute('COMMIT')

