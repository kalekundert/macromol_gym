import pymol
from pymol import cmd
from .database_io import open_db, select_zone_pdb_ids, select_zone_center_A

def show_zone(db_path, zone_id):
    db = open_db(db_path)
    zone_id = int(zone_id)

    pdb_ids = select_zone_pdb_ids(db, zone_id)
    center_A = select_zone_center_A(db, zone_id)

    cmd.set('assembly', pdb_ids['assembly_pdb_id'])
    cmd.fetch(pdb_ids['struct_pdb_id'])
    cmd.pseudoatom('zone', pos=list(center_A))

pymol.cmd.extend('show_zone', show_zone)


