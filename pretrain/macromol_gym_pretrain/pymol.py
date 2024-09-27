import pymol
import macromol_gym_pretrain as mmgp
import macromol_dataframe as mmdf
import numpy as np
import os
import re

from pymol import cmd
from pymol.wizard import Wizard
from macromol_gym_pretrain.database_io import (
        open_db, select_zone_ids, select_zone_pdb_ids, select_zone_atoms, 
        select_split,
)
from macromol_gym_pretrain.neighbors import (
        NeighborParams, get_neighboring_frames,
)
from macromol_gym_pretrain.geometry import cube_faces
from macromol_gym_pretrain.torch.infinite_sampler import InfiniteSampler
from macromol_voxelize import (
        ImageParams, Grid, 
        set_atom_radius_A, set_atom_channels_by_element,
)
from macromol_voxelize.pymol import (
        select_view, render_view, pick_channel_colors, cgo_cube_edges,
)
from macromol_dataframe import (
        make_coord_frame, invert_coord_frame, get_origin,
)
from contextlib import contextmanager
from pipeline_func import f
from math import pi

class TrainingExamples(Wizard):

    def __init__(
            self,
            db_path,
            length_voxels=24,
            resolution_A=1,
            atom_radius_A=None,
            channels=[['C'], ['N'], ['O'], ['*']],
            distance_A=30,
            noise_max_distance_A=2,
            noise_max_angle_deg=10,
            show_voxels=True,
            scale_alpha=False,
            split='train',
    ):
        super().__init__()

        length_voxels = int(length_voxels)
        resolution_A = float(resolution_A)
        atom_radius_A = float(atom_radius_A) if atom_radius_A else None
        distance_A = float(distance_A)
        noise_max_distance_A = float(noise_max_distance_A)
        noise_max_angle_deg = float(noise_max_angle_deg)

        self.db = open_db(db_path)
        self.zone_ids = select_split(self.db, split)
        self.neighbor_params = NeighborParams(
                direction_candidates=cube_faces(),
                distance_A=distance_A,
                noise_max_distance_A=noise_max_distance_A,
                noise_max_angle_deg=noise_max_angle_deg,
        )
        self.db_cache = {}
        self.img_params = ImageParams(
                grid=Grid(
                    length_voxels=length_voxels,
                    resolution_A=resolution_A,
                ),
                channels=len(channels),
        )
        self.atom_radius_A = atom_radius_A or resolution_A / 2
        self.channels = channels
        self.show_voxels = show_voxels
        self.scale_alpha = scale_alpha

        sampler = InfiniteSampler(
                len(self.zone_ids),
                shuffle=True,
        )
        self.zone_order = list(sampler)

        self.i = 0
        self.random_seed = 0

        self.redraw()

    def get_panel(self):
        panel = [
                [1, "Neighbor Dataset", ''],
                [2, "Next <C-Space>", 'cmd.get_wizard().next_training_example()'],
                [2, "Previous", 'cmd.get_wizard().prev_training_example()'],
                [2, "New random seed", 'cmd.get_wizard().new_random_seed()'],
                [3, f"Distance: {self.neighbor_params.distance_A}A", 'distance_A'],
                [3, f"Noise distance: {self.neighbor_params.noise_max_distance_A}A", 'noise_max_distance_A'],
                [3, f"Noise angle: {self.neighbor_params.noise_max_angle_deg} deg", 'noise_max_angle_deg'],
                [3, f"Show voxels: {'yes' if self.show_voxels else 'no'}", 'show_voxels'],
                [3, f"Scale alpha: {'yes' if self.scale_alpha else 'no'}", 'scale_alpha'],
                [2, "Done", 'cmd.set_wizard()'],
        ]
        return panel

    def get_menu(self, tag):
        menus = {
                'distance_A': [[2, 'Distance', '']],
                'noise_max_distance_A': [[2, 'Noise distance', '']],
                'noise_max_angle_deg': [[2, 'Noise angle', '']],
                'show_voxels': [
                    [2, 'Show voxels', ''],
                    [1, 'yes', 'cmd.get_wizard().set_show_voxels(True)'],
                    [1, 'no', 'cmd.get_wizard().set_show_voxels(False)'],
                ],
                'scale_alpha': [
                    [2, 'Scale alpha', ''],
                    [1, 'yes', 'cmd.get_wizard().set_scale_alpha(True)'],
                    [1, 'no', 'cmd.get_wizard().set_scale_alpha(False)'],
                ],
        }

        curr_dist_A = self.neighbor_params.distance_A
        curr_noise_max_dist_A = self.neighbor_params.noise_max_distance_A
        curr_noise_max_angle_deg = self.neighbor_params.noise_max_angle_deg

        for d in [-10, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 10]:
            menus['distance_A'] += [[
                1, f'{d:+}A',
                f'cmd.get_wizard().set_neighbor_distance_A({curr_dist_A + d})'
            ]]
            menus['noise_max_distance_A'] += [[
                1, f'{d:+}A',
                f'cmd.get_wizard().set_noise_distance_A({curr_noise_max_dist_A + d})'
            ]]
            menus['noise_max_angle_deg'] += [[
                1, f'{d:+} deg',
                f'cmd.get_wizard().set_noise_angle_deg({curr_noise_max_angle_deg + d})'
            ]]

        return menus[tag]

    def get_prompt(self):
        return [f"Zone: {self.curr_zone_id}"]

    def do_key(self, key, x, y, mod):
        # This is <Ctrl-Space>; see `wt_vs_mut` for details.
        if (key, mod) == (0, 2):
            self.next_training_example()
        else:
            return 0

        cmd.refresh_wizard()
        return 1

    def get_event_mask(self):
        return Wizard.event_mask_key

    def next_training_example(self):
        self.i += 1
        self.random_seed = 0
        self.redraw()

    def prev_training_example(self):
        self.i -= 1
        self.random_seed = 0
        self.redraw()

    def new_random_seed(self):
        self.random_seed += 1
        self.redraw(keep_view=True)

    def set_neighbor_distance_A(self, value):
        self.neighbor_params.distance_A = value
        self.redraw()

    def set_noise_distance_A(self, value):
        self.neighbor_params.noise_max_distance_A = value
        self.redraw()

    def set_noise_angle_deg(self, value):
        self.neighbor_params.noise_max_angle_deg = value
        self.redraw()

    def set_show_voxels(self, value):
        self.show_voxels = value
        self.redraw()

    def set_scale_alpha(self, value):
        self.scale_alpha = value
        self.redraw()

    def redraw(self, keep_view=False):
        if not keep_view:
            cmd.delete('all')

        # Get the next training example:
        zone_id, frame_ia, frame_ab, b = get_neighboring_frames(
                self.db,
                self.zone_order[self.i] + self.random_seed * len(self.zone_ids),
                self.zone_ids,
                self.neighbor_params,
                self.db_cache,
        )
        self.curr_zone_id = zone_id
        frame_ib = frame_ab @ frame_ia

        # Load the relevant structure:
        zone_pdb = select_zone_pdb_ids(self.db, zone_id)
        pdb_path = mmdf.get_pdb_path(
                os.environ['PDB_MMCIF'],
                zone_pdb['struct_pdb_id'],
        )

        if not keep_view:
            cmd.set('assembly', zone_pdb['assembly_pdb_id'])
            cmd.load(pdb_path, state=zone_pdb['model_pdb_id'])
            cmd.remove('hydro or resn hoh')
            cmd.util.cbc('elem C')

        curr_pdb_obj = zone_pdb['struct_pdb_id']

        # Render the two neighbors:
        select_view(
                name='sele_a',
                sele=curr_pdb_obj,
                grid=self.img_params.grid,
                frame_ix=frame_ia,
        )
        select_view(
                name='sele_b',
                sele=curr_pdb_obj,
                grid=self.img_params.grid,
                frame_ix=frame_ib,
        )

        atoms = (
                select_zone_atoms(self.db, zone_id)
                | f(set_atom_radius_A, self.atom_radius_A)
                | f(set_atom_channels_by_element, self.channels)
        )
        render_view(
                atoms_i=atoms,
                img_params=self.img_params,
                outline=(1, 1, 0),
                frame_ix=frame_ia,
                channel_colors=pick_channel_colors(
                    'sele_a',
                    self.channels,
                ),
                obj_names=dict(
                    voxels='voxels_a',
                    outline='outline_a',
                ),
                img=self.show_voxels,
                scale_alpha=self.scale_alpha,
        )
        render_view(
                atoms_i=atoms,
                img_params=self.img_params,
                outline=(0.4, 0.4, 0),
                frame_ix=frame_ib,
                channel_colors=pick_channel_colors(
                    'sele_b',
                    self.channels,
                ),
                obj_names=dict(
                    voxels='voxels_b',
                    outline='outline_b',
                ),
                img=self.show_voxels,
                scale_alpha=self.scale_alpha,
        )

        if self.show_voxels:
            cmd.show('sticks', 'byres (sele_a or sele_b)')

        if not keep_view:
            cmd.zoom('sele_a or sele_b', buffer=10)
            cmd.center('sele_a or sele_b')

def mmgp_training_examples(db_path, *args, **kwargs):
    kwargs.pop('_self', None)
    wizard = TrainingExamples(db_path, *args, **kwargs)
    cmd.set_wizard(wizard)

pymol.cmd.extend('mmgp_training_examples', mmgp_training_examples)

class ManualClassifier(Wizard):

    def __init__(
            self,
            db_path,
            view_size_A=24,
            neighbor_distance_A=30,
            noise_max_distance_A=2,
            noise_max_angle_deg=10,
            shuffle_seed=0,
            initial_zone_id=None,
    ):
        super().__init__()

        self.db = open_db(db_path)
        self.zone_ids = select_zone_ids(self.db)
        self.neighbor_params = params = NeighborParams(
                direction_candidates=cube_faces(),
                distance_A=neighbor_distance_A,
                noise_max_distance_A=noise_max_distance_A,
                noise_max_angle_deg=noise_max_angle_deg,
        )
        self.db_cache = {}

        self.frames_ac = [
                make_coord_frame(u * neighbor_distance_A)
                for u in params.direction_candidates
        ]
        self.frame_names = get_frame_names(params.direction_candidates)
        self.frame_order = ['+X', '+Y', '+Z', '-X', '-Y', '-Z']
        self.grid = Grid(
                length_voxels=view_size_A,
                resolution_A=1,
        )
        self.curr_pdb_obj = None

        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(self.zone_ids)

        if initial_zone_id is None:
            self.i = 0
        else:
            self.i = self.zone_ids.index(initial_zone_id)

        self.init_settings()
        self.init_view_boxes()
        self.init_curr_example()

    def get_panel(self):
        return [
                [1, "Manual Classifier", ''],
                [2, "Submit", 'cmd.get_wizard().submit_guess()'],
                [2, "Skip", 'cmd.get_wizard().skip_guess()'],
                [2, "Done", 'cmd.set_wizard()'],
        ]

    def get_prompt(self):
        return [self.frame_names[self.curr_b]]

    def do_key(self, key, x, y, mod):
        tab = (9, 0)
        ctrl_tab = (9, 2)

        def update_guess(step):
            curr_name = self.frame_names[self.curr_b]
            curr_name_i = self.frame_order.index(curr_name)
            next_name_i = (curr_name_i + 1) % len(self.frame_order)
            next_name = self.frame_order[next_name_i]
            next_b = self.frame_names.index(next_name)
            self.update_guess(next_b)

        if (key, mod) == tab:
            update_guess(1)
            return 1
        if (key, mod) == ctrl_tab:
            update_guess(-1)
            return 1

        return 0

    def get_event_mask(self):
        return Wizard.event_mask_key

    def init_settings(self):
        cmd.set('cartoon_gap_cutoff', 0)

    def init_view_boxes(self):
        cmd.delete('all')

        grid = self.grid
        dim_yellow = 0.4, 0.4, 0
        dim_red = 0.4, 0, 0
        dim_green = 0, 0.4, 0
        dim_blue = 0, 0, 0.4

        frame_colors = {
                '+X': dim_red,
                '+Y': dim_green,
                '+Z': dim_blue,
        }

        boxes = []
        boxes += cgo_cube_edges(grid.center_A, grid.length_A, dim_yellow)

        for i, frame_ac in enumerate(self.frames_ac):
            origin = get_origin(frame_ac)
            color = frame_colors.get(self.frame_names[i], dim_yellow)
            boxes += cgo_cube_edges(origin, grid.length_A, color)

        cmd.load_cgo(boxes, 'positions')
        
    def init_curr_example(self):
        zone_id, frame_ia, frame_ab, b = get_neighboring_frames(
                self.db,
                self.i,
                self.zone_ids,
                self.neighbor_params,
                self.db_cache,
        )
        frame_ib = frame_ab @ frame_ia

        self.curr_zone_id = zone_id
        self.curr_frame_ia = frame_ia
        self.curr_frame_ab = frame_ab
        self.curr_true_b = b

        cmd.delete(self.curr_pdb_obj)

        zone_pdb = select_zone_pdb_ids(self.db, zone_id)
        pdb_obj = self.curr_pdb_obj = zone_pdb['struct_pdb_id']
        pdb_path = mmdf.get_pdb_path(os.environ['PDB_MMCIF'], pdb_obj)

        cmd.set('assembly', zone_pdb['assembly_pdb_id'])
        cmd.load(pdb_path, state=zone_pdb['model_pdb_id'])
        cmd.disable(pdb_obj)

        select_view(
                name='sele_a',
                sele=pdb_obj,
                grid=self.grid,
                frame_ix=frame_ia,
        )
        select_view(
                name='sele_b',
                sele=pdb_obj,
                grid=self.grid,
                frame_ix=frame_ib,
        )

        cmd.delete('view_a')
        cmd.delete('view_b')

        cmd.create('view_a', 'sele_a')
        cmd.create('view_b', 'sele_b')

        cmd.delete('sele_a')
        cmd.delete('sele_b')

        frame_1d = list(frame_ia.flat)
        cmd.set_object_ttt('view_a', frame_1d)
        cmd.set_object_ttt(pdb_obj, frame_1d)

        self.update_guess(0)

        cmd.remove('hydro or resn hoh')
        cmd.util.cbag()
        cmd.util.cbac(pdb_obj)
        # cmd.hide('everything', pdb_obj)
        # cmd.show('cartoon', 'view_a or view_b')
        cmd.show('sticks', 'view_a or view_b')
        cmd.zoom('positions', buffer=5)

    def update_guess(self, c):
        # Picture of the relevant coordinate frames:
        # $ base64 -d - > frames.png <<< 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

        frame_ca = invert_coord_frame(self.frames_ac[c])
        frame_ia = frame_ca @ self.curr_frame_ab @ self.curr_frame_ia
        frame_1d = list(frame_ia.flat)
        cmd.set_object_ttt('view_b', frame_1d)
        
        self.curr_b = c
        cmd.refresh_wizard()

    def submit_guess(self):
        guess = self.frame_names[self.curr_b]
        answer = self.frame_names[self.curr_true_b]
        correct = (self.curr_b == self.curr_true_b)

        print(f"Zone id: {self.zone_ids[self.i]};  Guess: {guess};  Answer: {answer};  {'Correct' if correct else 'Incorrect'}!")

        self.i += 1
        self.init_curr_example()

    def skip_guess(self):
        answer = self.frame_names[self.curr_true_b]
        print(f"Zone id: {self.zone_ids[self.i]};  Guess: --;  Answer: {answer};  Skipped!")

        self.i += 1
        self.init_curr_example()

def mmgp_manual_classifier(db_path):
    wizard = ManualClassifier(db_path)
    cmd.set_wizard(wizard)

pymol.cmd.extend('mmgp_manual_classifier', mmgp_manual_classifier)

def get_frame_names(directions):
    # Currently, only "cube face" frames are supported.
    names_from_origins = {
            ( 1,  0,  0): '+X',
            (-1,  0,  0): '-X',
            ( 0,  1,  0): '+Y',
            ( 0, -1,  0): '-Y',
            ( 0,  0,  1): '+Z',
            ( 0,  0, -1): '-Z',
    }
    names = []

    for direction in directions:
        key = tuple(np.rint(direction).astype(int))
        name = names_from_origins[key]
        names.append(name)

    return names


def _sum_occ(sele):
    with sele_or_pseudoatom(sele) as sele:
        counter = {'q': 0}
        cmd.iterate(sele, 'counter["q"] += q', space=locals())
        return counter['q']

def sum_occ(sele):
    print(_sum_occ(sele), 'atoms')

pymol.cmd.extend('sum_occ', sum_occ)
cmd.auto_arg[0]['sum_occ'] = cmd.auto_arg[0]['zoom']

def _density(center, radius_A):
    with sele_or_pseudoatom(center) as sele:
        radius_nm = float(radius_A) / 10
        volume_nm3 = 4/3 * pi * radius_nm**3
        occupancy = _sum_occ(f'all within {radius_A} of ({sele})')
        return occupancy / volume_nm3

def density(center, radius_A):
    cmd.iterate_state(0, center, 'print(f"{x:.3f} {y:.3f} {z:.3f}")')
    print(_density(center, radius_A), 'atoms/nm^3')

pymol.cmd.extend('density', density)
cmd.auto_arg[0]['density'] = cmd.auto_arg[0]['zoom']

def show_centers(spacing_A=10, density_target_atoms_nm3=35, density_radius_A=15, sele='all', state=0):
    cmd.delete('zone_centers')

    spacing_A = float(spacing_A)
    density_target_atoms_nm3 = float(density_target_atoms_nm3)
    density_radius_A = float(density_radius_A)

    atoms = mmdf.from_pymol(sele, state)
    zone_centers_A = mmgp.calc_zone_centers_A(atoms, spacing_A)

    centers_above_density_target = 0

    for i, center_A in enumerate(zone_centers_A):
        density_atoms_nm3 = _density(center_A, density_radius_A)

        if density_atoms_nm3 >= density_target_atoms_nm3:
            centers_above_density_target += 1
            color = 'yellow'
            print(i, center_A)
        else:
            shade = 10 * int(10 * density_atoms_nm3 / density_target_atoms_nm3)
            color = f'gray{shade}'

        cmd.pseudoatom(
                'zone_centers',
                pos=repr(list(center_A)),
                resi=i,
                color=color,
        )

    print(f"Centers with >{density_target_atoms_nm3} atoms/nm^3: {centers_above_density_target}")

pymol.cmd.extend('show_centers', show_centers)

def show_neighbors(sele='sele', geometry='icosahedron faces', distance_A=30, density_target_atoms_nm3=35, density_radius_A=15, state=0):
    cmd.delete('neighbor_centers')

    distance_A = float(distance_A)
    density_target_atoms_nm3 = float(density_target_atoms_nm3)
    density_radius_A = float(density_radius_A)

    neighbors = mmgp.find_neighbor_centers_A(geometry, distance_A)

    center_A = np.zeros(3)
    cmd.iterate_state(state, sele, 'center_A[:] = (x, y, z)', space=locals())

    neighbors_above_density_target = 0

    for i, offset_A in enumerate(neighbors):
        neighbor_A = center_A + offset_A
        density_atoms_nm3 = _density(neighbor_A, density_radius_A)

        if density_atoms_nm3 >= density_target_atoms_nm3:
            neighbors_above_density_target += 1
            color = 'yellow'
            print(i, neighbor_A)
        else:
            shade = 10 * int(10 * density_atoms_nm3 / density_target_atoms_nm3)
            color = f'gray{shade}'

        cmd.pseudoatom(
                'neighbor_centers',
                pos=repr(list(neighbor_A)),
                resi=i,
                color=color,
        )

    print(f"Neighbors with >{density_target_atoms_nm3} atoms/nm^3: {neighbors_above_density_target}")

pymol.cmd.extend('show_neighbors', show_neighbors)
cmd.auto_arg[0]['show_neighbors'] = cmd.auto_arg[0]['zoom']

@contextmanager
def sele_or_pseudoatom(sele_or_xyz):
    if isinstance(sele_or_xyz, str):
        sele = sele_or_xyz
        yield sele
    else:
        xyz = sele_or_xyz
        with tmp_pseudoatom(xyz) as sele:
            yield sele

@contextmanager
def tmp_pseudoatom(xyz):
    name = '__xyz'
    names = cmd.get_names('all')

    if name in names:
        pattern = fr'^{name}(\d+)$'
        i = 1 + max(
                (m.group(1) for x in names if (m := re.match(pattern, x))),
                default=0,
        )
        name = f'{name}{i}'

    cmd.pseudoatom(name, pos=list(xyz))

    try:
        yield name
    finally:
        cmd.delete(name)


