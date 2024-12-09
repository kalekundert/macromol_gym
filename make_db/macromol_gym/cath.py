import polars as pl

from inform import parse_range
from pathlib import Path
from more_itertools import one

def load_cath_domains(cath_dir):
    rel_paths = [
            'cath-classification-data/cath-domain-list.txt',
            'cath-classification-data/cath-domain-boundaries-seqreschopping.txt',
    ]
    paths = [get_cath_path(cath_dir, p) for p in rel_paths]

    domain_list = parse_cath_domain_list(paths[0])
    domain_resis = parse_cath_domain_boundaries_seqres(paths[1])

    # In v4.3.0, the `domain_resis` table has ≈40k (≈8%) more rows than the 
    # `domain_list` table.  Every row in the `domain_list` table has a 
    # corresponding row in the `domain_resis` table.  In the few cases I looked 
    # at, the extra domains belong to structures that have other domains in 
    # `domain_list`.  I suspect that these extra domains were identified by the 
    # domain chopping algorithm, but not assigned to a CATH hierarchy.

    domains = domain_list.join(
            domain_resis,
            on=['pdb_id', 'chain_id', 'domain_id'],
            how='left',
            coalesce=True,
    )

    return domains, dict(zip(rel_paths, paths))

def parse_cath_domain_list(path):
    rows = []

    for line in Path(path).read_text().splitlines():
        if line.startswith('#'):
            continue

        name, c, a, t, h = line.split()[:5]
        assert len(name) == 7

        row = dict(
            pdb_id=name[0:4],
            chain_id=name[4:5],
            domain_id=int(name[5:7]),
            c=int(c),
            a=int(a),
            t=int(t),
            h=int(h),
        )
        rows.append(row)

    return pl.DataFrame(rows)

def parse_cath_domain_boundaries_seqres(path):
    rows = []

    for line in Path(path).read_text().splitlines():
        if line.startswith('#'):
            continue

        name, seq_ids = line.split() 
        assert len(name) == 7

        row = dict(
            pdb_id=name[0:4],
            chain_id=name[4:5],
            domain_id=int(name[5:7]),
            seq_ids=list(parse_range(seq_ids)),
        )
        rows.append(row)

    return pl.DataFrame(rows)

def get_cath_path(dir, name):
    path = dir / name
    if path.exists():
        return path

    name_glob = f'{path.stem}-v?_?_?{path.suffix}'
    path_glob = path.parent.glob(name_glob)

    return one(
            path_glob,
            too_short=ValueError(f"CATH file not found, expected to find one of:\n{path}\n{path.parent / name_glob}"),
            too_long=ValueError(f"Found multiple files matching:\n{path.parent / name_glob}"),
    )

