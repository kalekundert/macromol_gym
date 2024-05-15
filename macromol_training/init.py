"""
Create an empty database.

Usage:
    mmt_init <db>

Subsequent commands will fill this database with training examples.
"""

from .database_io import open_db, init_db

def main():
    import docopt
    args = docopt.docopt(__doc__)

    db = open_db(args['<db>'], mode='rwc')
    with db:
        init_db(db)
