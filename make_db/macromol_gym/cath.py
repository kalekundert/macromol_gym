import polars as pl
import os

from parsy import generate, regex, whitespace
from dataclasses import dataclass
from pathlib import Path

def parse_cath_chain_list(path):
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
            c=c,
            a=a,
            t=t,
            h=h,
        )
        rows.append(row)

    return pl.DataFrame(rows)

def parse_cath_domain_boundaries(path):
    domains = {}

    for line in Path(path).read_text().splitlines():
        if line.startswith('#'):
            continue

        chain = chain_spec.parse(line)

        for i, domain in enumerate(chain.domains, 0):
            k = f'{chain.pdb_id}{chain.chain_id}{i:02}'
            assert k not in domains
            domains[k] = domain

    return domains

def get_cath_path():
    return os.environ['CATH_PLUS']


@generate
def chain_spec():
    pdb_id = yield regex(r'\w{4}').desc('PDB id')
    chain_id = yield regex(r'[A-Z0]').desc('chain')
    yield whitespace

    domains = []
    fragments = []

    num_domains = yield regex(r'D(\d{2})', group=1).map(int).desc('num domains')
    yield whitespace
    num_fragments = yield regex(r'F(\d{2})', group=1).map(int).desc('num fragments')

    for i in range(num_domains):
        yield whitespace
        domain = yield domain_spec
        domains.append(domain)

    for i in range(num_fragments):
        yield whitespace
        fragment = yield fragment_spec
        fragments.append(fragment)

    return Chain(
            pdb_id=pdb_id,
            chain_id=chain_id,
            domains=domains,
            fragments=fragments,
    )

@generate
def domain_spec():
    segments = []
    num_segments = yield regex(r'\d+').map(int).desc('num segments')

    for i in range(num_segments):
        yield whitespace
        segment = yield segment_spec
        segments.append(segment)

    return segments

@generate
def segment_spec():
    start = yield residue_spec.desc('start residue')
    yield whitespace
    end = yield residue_spec.desc('end residue')

    return Segment(start, end)

@generate
def fragment_spec():
    segment = yield segment_spec
    yield whitespace

    # I don't know why this information (the number of residues) is provided; 
    # it seems 100% redundant...
    yield regex(r'[(]\d+[)]')

    return segment

@generate
def residue_spec():
    chain_id = yield regex(r'[A-Z]').desc('chain')
    yield whitespace
    index = yield regex(r'-?[0-9]+').desc('index')
    yield whitespace
    ins_code = yield regex(r'-|[A-Z]').desc('insertion code')

    if ins_code == '-':
        ins_code = ''

    return Residue(chain_id, index + ins_code)

@dataclass
class Residue:
    chain_id: str
    seq_label: str

@dataclass
class Segment:
    start: Residue
    end: Residue

    @classmethod
    def from_labels(cls, chain_id, start_label, end_label):
        start = Residue(chain_id, start_label)
        end = Residue(chain_id, end_label)
        return cls(start, end)

@dataclass
class Chain:
    pdb_id: str
    chain_id: str
    domains: list[list[Segment]]
    fragments: list[Segment]
