"""
Split the dataset such that structures from the same cluster always end up in 
the same split.

Usage:
    mmt_split_training_examples <db> <splits> <clusters> [-s <seed>] [-d]

Options:
    -s --random-seed <int>          [default: 0]
        Groups are assigned to splits with probability proportional to the 
        space that would be left in that split, after the group is added.  This 
        is a random but deterministic, controlled by this seed.

    -d --dry-run
        Calculate splits and report the final statistics, but don't write the 
        splits to the database.

It's ok to run this command multiple times.  Each invocation will overwrite any 
previously assigned splits.
"""

import networkx as nx
import polars as pl
import numpy as np
import nestedtext as nt

from .database_io import open_db, update_splits
from more_itertools import pairwise

def main():
    from docopt import docopt

    args = docopt(__doc__)

    db = open_db(args['<db>'], mode='rw')
    rng = np.random.default_rng(int(args['--random-seed']))
    struct_zone_counts = select_struct_zone_counts(db)
    targets = load_split_targets(args['<splits>'])
    clusters = load_clusters(args['<clusters>'])

    splits = make_splits(rng, struct_zone_counts, clusters, targets)
    summarize_splits(struct_zone_counts, targets, splits)

    if args['--dry-run']:
        print("\nDry run: the above splits were not saved")
        return

    with db:
        update_splits(db, splits)

def load_split_targets(split_path):
    return {k: float(v) for k, v in nt.load(split_path).items()}

def load_clusters(csv_path):
    return pl.read_csv(
            csv_path,
            new_columns=['pdb_id', 'cluster'],
            has_header=False,
    )

def make_splits(rng, struct_zone_counts, clusters, targets):
    n = sum(struct_zone_counts.values())
    m = sum(targets.values())

    curr_split_counts = {k: 0 for k in targets}
    target_split_counts = {k: n * v / m for k, v in targets.items()}

    groups = group_related_structures(
            struct_zone_counts.keys(),
            clusters,
    )

    def by_zone_count(group):
        return -sum(struct_zone_counts[x] for x in group), min(group)

    groups = sorted(groups, key=by_zone_count)
    splits = {}

    for group in groups:
        group_zone_count = sum(struct_zone_counts[x] for x in group)
        split_leftover_counts = {
                k: target_split_counts[k]
                   - curr_split_counts[k]
                   - group_zone_count
                for k in targets
        }
        if all(v <= 0 for v in split_leftover_counts.values()):
            split_leftover_counts = {
                    k: target_split_counts[k]
                       - curr_split_counts[k]
                    for k in targets
            }

        v_sum = sum(v for v in split_leftover_counts.values() if v > 0)
        weights = {
                k: v / v_sum
                for k, v in split_leftover_counts.items()
                if v > 0
        }
        assert weights

        split = rng.choice(list(weights.keys()), p=list(weights.values()))
        curr_split_counts[split] += group_zone_count

        for pdb_id in group:
            assert pdb_id not in splits
            splits[pdb_id] = split

    return splits

def group_related_structures(structures, clusters):
    g = nx.Graph()
    g.add_nodes_from(structures)

    for _, df in clusters.group_by(['cluster']):
        g.add_edges_from(pairwise(df['pdb_id']))

    def by_len(item):
        return len(item), min(item)

    return nx.connected_components(g)

def select_struct_zone_counts(db):
    cur = db.execute('''\
            SELECT 
                structure.pdb_id AS pdb_id,
                count(*) AS len
            FROM zone
            JOIN assembly ON assembly.id = zone.assembly_id
            JOIN structure ON structure.id = assembly.struct_id
            GROUP BY structure.pdb_id
    ''')
    return dict(cur.fetchall())

def summarize_splits(struct_zone_counts, targets, splits):
    def fraction_from_counts(col):
        return pl.col(col) / pl.col(col).sum()

    split_counts = (
            pl.DataFrame(
                list(splits.items()),
                ['pdb_id', 'split'],
                orient='row',
            )
            .with_columns(
                zone_count=pl.col('pdb_id').replace(
                    struct_zone_counts,
                    return_dtype=pl.Int32,
                ),
            )
            .group_by('split')
            .agg(
                pl.len().alias('struct_count'),
                pl.col('zone_count').sum()
            )
            .with_columns(
                struct_fraction=fraction_from_counts('struct_count'),
                zone_fraction=fraction_from_counts('zone_count'),
            )
            .rows_by_key('split', named=True, unique=True)
    )

    for k in targets:
        print(f"{k}:")
        print(f"- {split_counts[k]['struct_count']} ({100 * split_counts[k]['struct_fraction']:.2f}%) structures")
        print(f"- {split_counts[k]['zone_count']} ({100 * split_counts[k]['zone_fraction']:.2f}%) zones (target: {100 * targets[k] / sum(targets.values()):.2f}%)")
