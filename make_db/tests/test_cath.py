import macromol_gym.cath as _mmg
import parametrize_from_file as pff

with_cath = pff.Namespace('from macromol_gym.cath import *')

@pff.parametrize(
        indirect=['tmp_files']
)
def test_parse_cath_domain_boundaries(tmp_files, expected):
    expected = with_cath.eval(expected)

    path = tmp_files / 'cath-domain-boundaries.txt'
    domains = _mmg.parse_cath_domain_boundaries(path)

    assert domains == expected
