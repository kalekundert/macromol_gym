import macromol_training as mmt
import parametrize_from_file as pff
import polars as pl
import numpy as np

from pytest_unordered import unordered

class MockRng:

    def choice(self, items, *, p):
        i = np.argmax(p)
        return items[i]

def clusters(params):
    return pl.DataFrame(
            [x.split() for x in params.splitlines()],
            ['pdb_id', 'cluster'],
            orient='row',
    )

def int_dict(params):
    return {k: int(v) for k, v in params.items()}


@pff.parametrize(
        schema=pff.cast(clusters=clusters),
)
def test_group_related_structures(structures, clusters, expected):
    groups = mmt.group_related_structures(structures, clusters)

    def normalize_groups(groups):
        return [set(x) for x in groups]

    groups = normalize_groups(groups)
    expected = normalize_groups(groups)

    assert groups == unordered(expected)

@pff.parametrize(
        schema=pff.cast(
            struct_zone_counts=int_dict,
            clusters=clusters,
            targets=int_dict,
        ),
)
def test_make_splits(struct_zone_counts, clusters, targets, expected):
    rng = MockRng()
    splits = mmt.make_splits(rng, struct_zone_counts, clusters, targets)
    assert splits == expected
