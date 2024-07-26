import macromol_gym_unsupervised as mmgu

def test_copy_db_to_tmp(tmp_path):
    f = tmp_path / 'mock_db.sqlite'
    f.write_text('this is a database')

    with mmgu.copy_db_to_tmp(f) as tmp_f:
        assert tmp_f.exists()
        assert tmp_f.name == 'db.sqlite'
        assert tmp_f.read_text() == 'this is a database'

    assert f.exists()
    assert not tmp_f.exists()

