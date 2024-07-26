import logging
import tempfile
import shutil
import os

from contextlib import contextmanager
from pathlib import Path
from typing import Optional

log = logging.getLogger('macromol_gym')

@contextmanager
def copy_db_to_tmp(src_path, dest_name='db.sqlite', write=False, noop=False):
    if noop:
        yield Path(src_path)
        return

    with tempfile.TemporaryDirectory(prefix='macromol_gym_') as d:
        dest_path = Path(d) / dest_name
        log.info("copy database to local drive: src=%s dest=%s", src_path, dest_path)
        shutil.copy(src_path, dest_path)

        if not write:
            dest_path.chmod(0o444)

        yield dest_path

        if write:
            shutil.copy(dest_path, src_path)

def get_num_workers(num_workers: Optional[int]) -> int:
    if num_workers is not None:
        return num_workers

    try:
        return int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    except KeyError:
        return os.cpu_count()

