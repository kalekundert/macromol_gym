import macromol_gym.cath as _mmg
import parametrize_from_file as pff
import polars as pl
import polars.testing

from macromol_dataframe.testing import dataframe

@pff.parametrize(
        schema=pff.cast(
            expected=dataframe(
                dtypes=dict(
                    domain_id=int,
                    c=int,
                    a=int,
                    t=int,
                    h=int,
                ),
            )
        ),
        indirect=['tmp_files'],
)
def test_parse_cath_domain_list(tmp_files, expected):
    path = tmp_files / 'cath-domain-list.txt'
    domains = _mmg.parse_cath_domain_list(path)

    pl.testing.assert_frame_equal(domains, expected)

@pff.parametrize(
        schema=pff.cast(
            expected=dataframe(
                dtypes=dict(domain_id=int),
                exprs=dict(
                    seq_ids=pl.col('seq_ids').str.split(',').cast(list[int])
                ),
            )
        ),
        indirect=['tmp_files'],
)
def test_parse_cath_domain_boundaries_seqres(tmp_files, expected):
    path = tmp_files / 'cath-domain-boundaries-seqreschopping.txt'
    df = _mmg.parse_cath_domain_boundaries_seqres(path)

    pl.testing.assert_frame_equal(df, expected)



