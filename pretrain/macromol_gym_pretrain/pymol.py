import pymol
import macromol_dataframe as mmdf
import polars as pl
import numpy as np
import nestedtext as nt
import os

from pymol import cmd
from pymol.wizard import Wizard
from torch_deterministic import InfiniteSampler
from macromol_gym_pretrain.torch import (
        MacromolDataset, ImageParams, NeighborParams,
        make_neighbor_sample, make_neighbor_image_sample,
        open_db, select_zone_pdb_ids, polyhedron_faces, cube_faces,
)
from macromol_voxelize import Grid
from macromol_voxelize.pymol import (
        select_view, render_image, pick_channel_colors, cgo_cube_edges,
)
from macromol_dataframe import make_coord_frame, invert_coord_frame, get_origin
from voluptuous import Schema, Coerce
from platformdirs import user_config_path
from collections import ChainMap
from functools import partial
from itertools import count
from pathlib import Path
from csv import DictWriter

NEIGHBOR_GEOMETRIES = {
        4: 'tetrahedron',
        6: 'cube',
        8: 'octahedron',
        12: 'dodecahedron',
        20: 'icosahedron',
}

class TrainingExamples(Wizard):
    DEFAULT_CONFIG_PATH = user_config_path('macromol_gym') / 'mmgp_training_examples.nt'

    def __init__(
            self,
            db_path=None,
            config_path=None,
    ):
        super().__init__()

        self.dialog_prompt = None
        self.dialog_callback = None
        self.dialog_input = ''
        self.dialog_error = ''

        self.load_config(db_path, config_path)
        self.load_dataset()

    def get_panel(self):
        panel = [
                [1, "Neighbor Dataset", ''],
                [2, "Next <C-Space>", 'cmd.get_wizard().next_training_example()'],
                [2, "Previous", 'cmd.get_wizard().prev_training_example()'],
                [2, "New random seed", 'cmd.get_wizard().new_random_seed()'],
                [2, "Done", 'cmd.set_wizard()'],

                [1, "\\555Parameters", ''],
                [2, f"Image length: \\090{self.img_params.grid.length_voxels} voxels", 'cmd.get_wizard().start_image_length_dialog()'],
                [2, f"Image resolution: \\090{self.img_params.grid.resolution_A}A", 'cmd.get_wizard().start_image_resolution_dialog()'],
                [2, f"Atom radius: \\090{self.img_params.atom_radius_A}A", 'cmd.get_wizard().start_atom_radius_dialog()'],
                [3, f"Neighbor geometry: {NEIGHBOR_GEOMETRIES[len(self.neighbor_params.direction_candidates)]}", 'neighbor_geometry'],
                [2, f"Neighbor distance: \\090{self.neighbor_params.distance_A}A", 'cmd.get_wizard().start_neighbor_distance_dialog()'],
                [2, f"Noise distance: \\090{self.neighbor_params.noise_max_distance_A}A", 'cmd.get_wizard().start_noise_max_distance_dialog()'],
                [2, f"Noise angle: \\090{self.neighbor_params.noise_max_angle_deg} deg", 'cmd.get_wizard().start_noise_max_angle_deg_dialog()'],
                [3, f"Show voxels: {'yes' if self.show_voxels else 'no'}", 'show_voxels'],
                [3, f"Scale alpha: {'yes' if self.scale_alpha else 'no'}", 'scale_alpha'],
        ]
        return panel

    def get_menu(self, tag):
        menus = {
                'neighbor_geometry': [
                    [2, 'Neighbor geometry', ''],
                    [1, 'tetrahedron', 'cmd.get_wizard().set_neighbor_geometry("tetrahedron")'],
                    [1, 'cube', 'cmd.get_wizard().set_neighbor_geometry("cube")'],
                    [1, 'octahedron', 'cmd.get_wizard().set_neighbor_geometry("octahedron")'],
                    [1, 'dodecahedron', 'cmd.get_wizard().set_neighbor_geometry("dodecahedron")'],
                    [1, 'icosahedron', 'cmd.get_wizard().set_neighbor_geometry("icosahedron")'],
                ],
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
        return menus[tag]

    def get_prompt(self):
        if self.dialog_prompt is None:
            return [f"Zone: {self.curr_zone_id}"]

        prompt = [f"{self.dialog_prompt} \\999{self.dialog_input}"]
        if self.dialog_error:
            prompt += [f"\\900Error: {self.dialog_error}"]

        return prompt

    def do_key(self, key, x, y, mod):
        ESC = (27, 0)
        BACKSPACE = (8, 0)
        ENTER = (10, 13)
        CTRL_SPACE = (0, 2)

        if self.dialog_prompt is not None:

            if (key, mod) == ESC:
                self.dialog_prompt = None
                self.dialog_callback = None
                self.dialog_input = ''
                self.dialog_error = ''
                self.redraw()

            elif (key, mod) == BACKSPACE:
                self.dialog_input = self.dialog_input[:-1]

            elif key >= 32:
                self.dialog_input += chr(key)

            elif key in ENTER:
                try:
                    self.dialog_callback(self.dialog_input)
                except Exception as err:
                    self.dialog_error = str(err)
                    self.redraw()
                else:
                    self.dialog_prompt = None
                    self.dialog_callback = None
                    self.dialog_input = ''
                    self.dialog_error = ''
                    self.redraw()

            else:
                return 0

        elif (key, mod) == CTRL_SPACE:
            self.next_training_example()

        else:
            return 0

        cmd.refresh_wizard()
        return 1

    def get_event_mask(self):
        return Wizard.event_mask_key


    def load_config(self, db_path, config_path):
        schema = Schema({
            'db_path': str,
            'db_split': str,
            'image_length_voxels': Coerce(int),
            'image_resolution_A': Coerce(float),
            'atom_radius_A': Coerce(float),
            'element_channels': [[str]],
            'neighbor_geometry': str,
            'neighbor_distance_A': Coerce(float),
            'noise_max_distance_A': Coerce(float),
            'noise_max_angle_deg': Coerce(float),
            'show_voxels': Coerce(bool),
            'scale_alpha': Coerce(bool),
        })

        config_chain = ChainMap()

        if db_path is not None:
            config_chain.maps.append({'db_path': db_path})

        if config_path is not None:
            config_i = schema(nt.load(config_path))
            config_chain.maps.append(config_i)

        if self.DEFAULT_CONFIG_PATH.exists():
            config_i = schema(nt.load(self.DEFAULT_CONFIG_PATH))
            config_chain.maps.append(config_i)

        config_chain.maps.append({
            'db_split': 'train',
            'image_length_voxels': 24,
            'image_resolution_A': 1.0,
            'element_channels': [['C'], ['N'], ['O'], ['*']],
            'neighbor_geometry': 'cube',
            'noise_max_distance_A': 0,
            'noise_max_angle_deg': 0,
            'show_voxels': True,
            'scale_alpha': False,
        })

        self.db_path = Path(config_chain['db_path'])
        self.db_split = config_chain['db_split']

        grid = Grid(
                length_voxels=config_chain['image_length_voxels'],
                resolution_A=config_chain['image_resolution_A'],
        )
        self.img_params = ImageParams(
            grid=grid,
            atom_radius_A=config_chain.get('atom_radius_A', grid.resolution_A / 2),
            element_channels=config_chain['element_channels'],
        )
        self.neighbor_params = NeighborParams(
            direction_candidates=polyhedron_faces(config_chain['neighbor_geometry']),
            distance_A=config_chain.get('neighbor_distance_A', grid.length_A),
            noise_max_distance_A=config_chain['noise_max_distance_A'],
            noise_max_angle_deg=config_chain['noise_max_angle_deg'],
        )
        self.show_voxels = config_chain['show_voxels']
        self.scale_alpha = config_chain['scale_alpha']

    def load_dataset(self):
        self.db = open_db(self.db_path)
        self.db_cache = {}

        self.dataset = MacromolDataset(
                db_path=self.db_path,
                split=self.db_split,
                make_sample=partial(
                    make_neighbor_image_sample,
                    img_params=self.img_params,
                    neighbor_params=self.neighbor_params,
                ),
        )
        self.sampler = InfiniteSampler(
                len(self.dataset),
                shuffle=True,
        )
        self.curr_permut = list(self.sampler)
        self.i = 0
        self.random_seed = 0

        self.redraw()

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

    def start_image_length_dialog(self):

        def set_image_length(x):
            self.img_params.grid = Grid(
                    length_voxels=int(x),
                    resolution_A=self.img_params.grid.resolution_A,
            )

        self.dialog_prompt = "Image length (voxels):"
        self.dialog_callback = set_image_length

        cmd.refresh_wizard()

    def start_image_resolution_dialog(self):

        def set_image_resolution(x):
            self.img_params.grid = Grid(
                    length_voxels=self.img_params.grid.length_voxels,
                    resolution_A=float(x),
            )

        self.dialog_prompt = "Image resolution (A):"
        self.dialog_callback = set_image_resolution

        cmd.refresh_wizard()

    def start_atom_radius_dialog(self):

        def set_atom_radius(x):
            self.img_params.atom_radius_A = float(x)

        self.dialog_prompt = "Atom radius (A):"
        self.dialog_callback = set_atom_radius

        cmd.refresh_wizard()

    def start_neighbor_distance_dialog(self):

        def set_neighbor_distance(x):
            self.neighbor_params.distance_A = float(x)

        self.dialog_prompt = "Neighbor distance (A):"
        self.dialog_callback = set_neighbor_distance

        cmd.refresh_wizard()

    def start_noise_max_distance_dialog(self):

        def set_noise_max_dist(x):
            self.neighbor_params.noise_max_distance_A = float(x)

        self.dialog_prompt = "Noise max distance (A):"
        self.dialog_callback = set_noise_max_dist

        cmd.refresh_wizard()

    def start_noise_max_angle_dialog(self):

        def set_noise_max_angle(x):
            self.neighbor_params.noise_max_angle_deg = float(x)

        self.dialog_prompt = "Noise max angle (deg):"
        self.dialog_callback = set_noise_max_angle

        cmd.refresh_wizard()

    def set_neighbor_geometry(self, value):
        self.neighbor_params.direction_candidates = polyhedron_faces(value)
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
        n = len(self.dataset)
        curr_epoch = self.i // n

        if curr_epoch != self.sampler.curr_epoch:
            self.sampler.set_epoch(curr_epoch)
            self.curr_permut = list(self.sampler)

        i = self.curr_permut[self.i % n] + n * self.random_seed
        x = self.dataset[i]

        self.curr_zone_id = x['zone_id']

        frame_ia, frame_ab = x['frame_ia'], x['frame_ab']
        frame_ib = frame_ab @ frame_ia

        # Load the relevant structure:
        zone_pdb = select_zone_pdb_ids(self.db, self.curr_zone_id)
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

        render_image(
                img=x['img_a'] if self.show_voxels else None,
                grid=self.img_params.grid,
                frame_xi=mmdf.invert_coord_frame(frame_ia),
                obj_names=dict(
                    voxels='voxels_a',
                    outline='outline_a',
                ),
                channel_colors=pick_channel_colors('sele_a', self.img_params.element_channels),
                outline=(1, 1, 0),
                scale_alpha=self.scale_alpha,
        )
        render_image(
                img=x['img_b'] if self.show_voxels else None,
                grid=self.img_params.grid,
                frame_xi=mmdf.invert_coord_frame(frame_ib),
                obj_names=dict(
                    voxels='voxels_b',
                    outline='outline_b',
                ),
                channel_colors=pick_channel_colors('sele_b', self.img_params.element_channels),
                outline=(0.4, 0.4, 0),
                scale_alpha=self.scale_alpha,
        )

        if self.show_voxels:
            cmd.show('sticks', 'byres (sele_a or sele_b)')

        if not keep_view:
            cmd.zoom('sele_a or sele_b', buffer=10)
            cmd.center('sele_a or sele_b')

def mmgp_training_examples(*args, **kwargs):
    """
    DESCRIPTION

        Visualize samples from the neighbor location dataset.

    USAGE

        mmgp_training_examples [db_path [, config_path]]

    ARGUMENTS

        db_path = str: Path to the database file.

        config_path = str: Path to a file specifying various dataset parameters.
        This should be a NestedText file with the following fields:

            db_path: Database to use (overridden by above argument)
            db_split: Which split of the data to use; defaults to 'train'
            image_length_voxels: Length of the image in voxels
            image_resolution_A: Length of each voxel in angstroms
            atom_radius_A: Radius of each atom in angstroms
            element_channels: List of lists of element symbols to include in each channel.
            neighbor_geometry: Possible neighbor locations.  Must be one of:
                'tetrahedron', 'cube', 'octagon', 'dodecahedron', 'icosahedron'
            neighbor_distance_A: Distance between image centers in angstroms
            noise_max_distance_A: Maximum amount of noise to add to the distance
            noise_max_angle_deg: Maximum amount to rotate the neighboring image
            show_voxels: Whether to show the images; 'yes' or 'no'
            scale_alpha: Whether to scale the alpha channel; 'yes' or 'no'

        Any values specified by this file will override any values specified in 
        the following default configuration file, if it exists:

            {DEFAULT_CONFIG_PATH}
            
    ENVIRONMENT VARIABLES

        PDB_MMCIF = str: Path to the directory containing mmCIF files for every 
        structure in the database, organized as in the PDB (e.g. `ab/1abc.cif`).

    HOTKEYS:

        Ctrl+Space: Advance to the next training example.

    SEE ALSO

        mmgp_manual_classifier, mmgu_training_examples, voxelize, load_voxels
    """
    kwargs.pop('_self', None)
    wizard = TrainingExamples(*args, **kwargs)
    cmd.set_wizard(wizard)

mmgp_training_examples.__doc__ = mmgp_training_examples.__doc__.format(
        DEFAULT_CONFIG_PATH=TrainingExamples.DEFAULT_CONFIG_PATH,
)
pymol.cmd.extend('mmgp_training_examples', mmgp_training_examples)

class ManualClassifier(Wizard):
    DEFAULT_CONFIG_PATH = user_config_path('macromol_gym') / 'mmgp_manual_classifier.nt'

    def __init__(
            self,
            db_path=None,
            config_path=None,
            log_path='mmgp_manual_classifier_log.csv',
    ):
        super().__init__()

        self.log_path = Path(log_path)

        self.load_config(db_path, config_path)
        self.load_dataset()
        self.load_frames()

        self.dialog_prompt = None
        self.dialog_callback = None
        self.dialog_input = ''
        self.dialog_error = ''

        cmd.delete('all')
        cmd.set('cartoon_gap_cutoff', 0)

        self.curr_pdb_obj = None

        self.render_curr_example()

    def get_panel(self):
        panel = [
                [1, "Manual Classifier", ''],
                [2, "Submit", 'cmd.get_wizard().submit_guess()'],
                [2, "Skip", 'cmd.get_wizard().skip_guess()'],
                [2, "Done", 'cmd.set_wizard()'],

                [1, "\\555Parameters", ''],
                [2, f"View size: \\090{self.view_size_A}A", 'cmd.get_wizard().start_view_size_dialog()'],
                [2, f"Neighbor distance: \\090{self.neighbor_params.distance_A}A", 'cmd.get_wizard().start_neighbor_distance_dialog()'],
                [2, f"Noise distance: \\090{self.neighbor_params.noise_max_distance_A}A", 'cmd.get_wizard().start_noise_max_distance_dialog()'],
                [2, f"Noise angle: \\090{self.neighbor_params.noise_max_angle_deg} deg", 'cmd.get_wizard().start_noise_max_angle_dialog()'],

                [1, "\\555Neighbor Toggles", ''],

        ]
        
        for name in self.frame_order:
            i = self.frame_names.index(name)

            status = '\\090maybe' if self.frame_toggles[i] else '\\900no   '
            if i == self.curr_b:
                status += '  \\999<--'

            panel.append(
                [2, f"{name}: {status}", f'cmd.get_wizard().toggle_frame({i})']
            )

        return panel

    def get_prompt(self):
        if self.dialog_prompt is None:
            return [self.frame_names[self.curr_b]]

        prompt = [f"{self.dialog_prompt} \\999{self.dialog_input}"]
        if self.dialog_error:
            prompt += [f"\\900Error: {self.dialog_error}"]

        return prompt

    def do_key(self, key, x, y, mod):
        if self.dialog_prompt is not None:
            return self.do_key_prompt(key, x, y, mod)
        else:
            return self.do_key_main(key, x, y, mod)

    def do_key_main(self, key, x, y, mod):
        TAB = (9, 0)
        CTRL_TAB = (9, 2)

        if (key, mod) == TAB:
            self.cycle_guess(1)
            return 1

        if (key, mod) == CTRL_TAB:
            self.cycle_guess(-1)
            return 1

        return 0

    def do_key_prompt(self, key, x, y, mod):
        ESC = (27, 0)
        BACKSPACE = (8, 0)
        ENTER = (10, 13)

        if (key, mod) == ESC:
            self.dialog_prompt = None
            self.dialog_callback = None
            self.dialog_input = ''
            self.dialog_error = ''

        elif (key, mod) == BACKSPACE:
            self.dialog_input = self.dialog_input[:-1]

        elif key >= 32:
            self.dialog_input += chr(key)

        elif key in ENTER:
            try:
                self.dialog_callback(self.dialog_input)
            except Exception as err:
                self.dialog_error = str(err)
            else:
                self.dialog_prompt = None
                self.dialog_callback = None
                self.dialog_input = ''
                self.dialog_error = ''

                self.render_curr_example()

        else:
            return 0

        cmd.refresh_wizard()
        return 1

    def get_event_mask(self):
        return Wizard.event_mask_key


    def load_config(self, db_path, config_path):
        schema = Schema({
            'db_path': str,
            'db_split': str,
            'view_size_A': Coerce(float),
            'neighbor_distance_A': Coerce(float),
            'noise_max_distance_A': Coerce(float),
            'noise_max_angle_deg': Coerce(float),
        })

        config_chain = ChainMap()

        if db_path is not None:
            config_chain.maps.append({'db_path': db_path})

        if config_path is not None:
            config_i = schema(nt.load(config_path))
            config_chain.maps.append(config_i)

        if self.DEFAULT_CONFIG_PATH.exists():
            config_i = schema(nt.load(self.DEFAULT_CONFIG_PATH))
            config_chain.maps.append(config_i)

        config_chain.maps.append({
            'db_split': 'train',
            'view_size_A': 15,
            'neighbor_distance_A': 18,
            'noise_max_distance_A': 0,
            'noise_max_angle_deg': 0,
        })

        self.db_path = Path(config_chain['db_path'])
        self.db_split = config_chain['db_split']

        self.view_size_A = config_chain['view_size_A']
        self.neighbor_params = NeighborParams(
            direction_candidates=cube_faces(),
            distance_A=config_chain['neighbor_distance_A'],
            noise_max_distance_A=config_chain['noise_max_distance_A'],
            noise_max_angle_deg=config_chain['noise_max_angle_deg'],
        )

    def load_dataset(self):
        self.db = open_db(self.db_path)
        self.db_cache = {}

        self.dataset = MacromolDataset(
                db_path=self.db_path,
                split=self.db_split,
                make_sample=partial(
                    make_neighbor_sample,
                    neighbor_params=self.neighbor_params,
                ),
        )
        self.sampler = InfiniteSampler(
                len(self.dataset),
                shuffle=True,
        )
        self.curr_permut = list(self.sampler)

        if self.log_path.exists():
            df = pl.read_csv(self.log_path)
            self.i = df['i'].max() + 1
        else:
            self.i = 0

    def load_frames(self):
        self.frames_ac = [
                make_coord_frame(u * self.neighbor_params.distance_A)
                for u in self.neighbor_params.direction_candidates
        ]
        self.frame_names = get_frame_names(self.neighbor_params.direction_candidates)
        self.frame_order = ['+X', '+Y', '+Z', '-X', '-Y', '-Z']
        self.frame_toggles = [True] * len(self.frames_ac)

    def render_view_boxes(self, keep_view=False):
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
        boxes += cgo_cube_edges(np.zeros(3), self.view_size_A, dim_yellow)

        for i, frame_ac in enumerate(self.frames_ac):
            if self.frame_toggles[i]:
                origin = get_origin(frame_ac)
                color = frame_colors.get(self.frame_names[i], dim_yellow)
                boxes += cgo_cube_edges(origin, self.view_size_A, color)

        if keep_view:
            view = cmd.get_view()

        cmd.delete('positions')
        cmd.load_cgo(boxes, 'positions')

        if keep_view:
            cmd.set_view(view)
        
    def render_curr_example(self):
        self.frame_toggles = [True] * len(self.frames_ac)
        self.render_view_boxes()

        n = len(self.dataset)
        curr_epoch = self.i // n

        if curr_epoch != self.sampler.curr_epoch:
            self.sampler.set_epoch(curr_epoch)
            self.curr_permut = list(self.sampler)

        i = self.curr_permut[self.i % n]
        x = self.dataset[i]

        frame_ia, frame_ab = x['frame_ia'], x['frame_ab']
        frame_ib = frame_ab @ frame_ia

        self.curr_zone_id = x['zone_id']
        self.curr_frame_ia = frame_ia
        self.curr_frame_ab = frame_ab
        self.curr_true_b = x['b']

        cmd.delete(self.curr_pdb_obj)

        zone_pdb = select_zone_pdb_ids(self.db, self.curr_zone_id)
        pdb_obj = self.curr_pdb_obj = zone_pdb['struct_pdb_id']
        pdb_path = mmdf.get_pdb_path(os.environ['PDB_MMCIF'], pdb_obj)

        cmd.set('assembly', zone_pdb['assembly_pdb_id'])
        cmd.load(pdb_path, state=zone_pdb['model_pdb_id'])
        cmd.disable(pdb_obj)

        grid = Grid(
                length_voxels=1,
                resolution_A=self.view_size_A,
        )
        select_view(
                name='sele_a',
                sele=pdb_obj,
                grid=grid,
                frame_ix=frame_ia,
        )
        select_view(
                name='sele_b',
                sele=pdb_obj,
                grid=grid,
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

        b = self.frame_names.index(self.frame_order[0])
        self.update_guess(b)

        cmd.remove('hydro or resn hoh')
        cmd.util.cbag()
        cmd.util.cbac(pdb_obj)
        cmd.show('cartoon', 'view_a or view_b')
        cmd.show('sticks', 'view_a or view_b')
        cmd.zoom('positions', buffer=5)

    def toggle_frame(self, i):
        self.frame_toggles[i] = not self.frame_toggles[i]
        if i == self.curr_b:
            self.cycle_guess(1)

        self.render_view_boxes(keep_view=True)
        cmd.refresh_wizard()

    def cycle_guess(self, step):
        curr_name = self.frame_names[self.curr_b]
        curr_name_i = self.frame_order.index(curr_name)

        for i in count(1):
            next_name_i = (curr_name_i + i * step) % len(self.frame_order)
            next_name = self.frame_order[next_name_i]
            next_b = self.frame_names.index(next_name)

            if self.frame_toggles[next_b]:
                break
            if next_b == self.curr_b:
                break

        self.update_guess(next_b)

    def update_guess(self, c):
        # Picture of the relevant coordinate frames:
        # $ base64 -d - > frames.png <<< iVBORw0KGgoAAAANSUhEUgAAAMgAAAChAgMAAAD/zRifAAAADFBMVEUeHh5iYmKmpqb5+fkKfYLFAAAGHElEQVRo3u2Zu28cRRjAF4mEwgUVTSLhFojE9RQkLghSqgjpEhEXboMjsX8BWSlCQSKFY5Q4UihOiu6kZFbc1hwiU1K4SIGiSFwRYjs+sitlheMzt7O+b5jX7s2eZx8DSgWj0/q8t7+b+d7fzDnUejj/I/8WCQNK2QvT1GoWzC8pjZohgK0XVkDw60IKj3n2Sq6ZhdgjmSzBP7M+QQjZOoyaKggJDoLahYnLxTlZXga4BknumcTHOChHyASXaizqG5Go37JRMr90gqGNw4hHO9QWoZ2Crm2Q2B6hrw0RIq/ZIhF9ROmmVex3KDNL2BxhL48jSctC/CnmCL1hoWRYdUekt7xtY8p1/LvTvrhnZ33ClDCyQ/gYVohP0sAgKHQaaQyQP3vv2vgYQX6IKLYwpQxxce0PEPJ91Az5rng7RbM1lyE9YyCzNEqM1udPEre0opkQWSltopKU1Y5yREwwtUcqKrMhKqMcaVpef4tyHTTWGEfIA+6pbmMEfApTisvEKZkl94JmeSySdmnuyQ99iRzgoCkC6woJrjZFyKpEiH9h/tnnsRnxpSwGtyTLkVn8oAyB63hUqeQjCFnCdM8KSY4zpR9oSEohzo03kmUyRJqW4Sw3U2KahQBC91ifBKjXbjsLWppWNcG0sMftM5+0z795DAWAzgUgF7gjHXTqmWUJe2xBqFAyeyf8hzwpQcvsYwkJsqBUCDmbfbhCfISO2gWwlzmyyuQbuFDcjyAuvV18giYzn14rQWCuY/h+ruYYkLkmQ5ukAcLfw7J2Y9QAwYVJpJNVIi2G+PqHB/VI8KAYzkkdAh/HkVdomia1yEc+xYWO6RCXIapSgEeAbs+1NyXILU8h7PJKT4TijgmBL1Zm3ymUNFHLhdMlyHRwRyGsnHYxC5n9c7nazciYjGUQ8l3M4Sq77HdrEJLItf/hikhksRN6NYhKC3BZLT+gw4ZIOsm95mlmLa4Xpx9BbEakrVnwshTV0WKs2Fxh0mchhguI8JhH6onNsoWNKaQsld33kT/AN/jdJ6pxflIti5rlF5ZCfmLpsPsBS4kbLP3WIyISx7Jxpr9WzLLL7SJubCsXvrOp/ilDpi4l92fBC6d3WmtVyGF7Nf6GxnJhY3F9f0Hkmz0jkvJO9tbiFg0mSiLucNdke2JEvrwt/+7SodTC4UDblplnuSu8GBborpzl0J1LMc78xu5z7Kkye9Bl1Zmgnz0D8pl278UiP7H4XnNLEYnZmYxC4LKXbzxJ0GHhQe5opqSnc4PmyP5ApF34QTQKXJUHuwpJsxihe5GOPEu2RPPCskMos/1oB0uEpCpGaPI11pBrk4TXrYck4GcdvB78eUk8NsHTjTSW3n/4YSvLsAz5dCLEuOpTpEoIwbF0rOlX8V0ZK/BuS36LQDqJKjeiLjzR0ymshtmm9DrOMqwz83VZF0ZaOvWEk4z0uPUKCMFHkdz7M6QlkQN9l6M9II2o+iN5Z6OAqB53u1DFMi3JwgQ3r6zpiDSfzPQFZJIjN69wXfBNIm+NQnQXo5hVha0QiUGh5+K8Dol3OyeFDZxsLVnlfe68w/qq9hnHaV/iDTDzilMsJDDv4mYd7EhzVfpMleAwL8VwijVOYYiW5PQoR4Lyo7WWUE3X1frkqObwcYUXvu5JOodU7L2YXwLtnih040NNFNPYpHD/BD2CVB3xbdPJ8bme/2nNOeoeLNGjSJUodLw+MCBxFTJxYR4ZVosyr09nFo8VwzBLSq0RaotE9sio6mSvEjllgahwX7dAXrG2i60qOdkcGV9AaOn8MfcZboxwQ7INVQdajZHiXqP2YN0BnLskLDZDtBR8PqBhH2Wjz/ZjQGPy4keMgbdcrGkWHq8vzDd7MYm4mWeHX/W/v6R/Pcbs22HBAuHHSwxZws2RQOyIp9+uNEZEpUopuRE3RtS+e2uT2iGEpkNLBLQi2hTBPETAt0CYx7CKCG/YIC9l++JZiL8rkDNvW8iyLJD9BWrzGx8v/8/fskWIYzcL65uWeotWyBTvuPueFQLe2ZKdePl4r1VyqlA+rqliZ4FYxovmbIHDWqHYB+Qj4xjk71jPFaAwAD/+j/9a/TdOSSfS2wkWbQAAAABJRU5ErkJggg==

        frame_ca = invert_coord_frame(self.frames_ac[c])
        frame_ia = frame_ca @ self.curr_frame_ab @ self.curr_frame_ia
        frame_1d = list(frame_ia.flat)
        cmd.set_object_ttt('view_b', frame_1d)
        
        self.curr_b = c
        cmd.refresh_wizard()

    def submit_guess(self):
        self.log_guess(skip=False)

        self.i += 1
        self.render_curr_example()

    def skip_guess(self):
        self.log_guess(skip=True)

        self.i += 1
        self.render_curr_example()

    def log_guess(self, skip):
        # Display a message in the console for the user to see immediately:
        answer = self.frame_names[self.curr_true_b]

        if skip:
            guess = '--'
            result = 'Skipped'
        else:
            guess = self.frame_names[self.curr_b]
            correct = (self.curr_b == self.curr_true_b)
            result = 'Correct' if correct else 'Incorrect'

        print(f"Zone id: {self.curr_zone_id};  Guess: {guess};  Answer: {answer};  {result}!")

        # Write the results to a CSV file for later analysis:
        row = dict(
                db_path=self.db_path.resolve(),
                db_split=self.db_split,
                i=self.i,
                zone_id=self.curr_zone_id,
                view_size_A=self.view_size_A,
                neighbor_distance_A=self.neighbor_params.distance_A,
                noise_max_distance_A=self.neighbor_params.noise_max_distance_A,
                noise_max_angle_deg=self.neighbor_params.noise_max_angle_deg,
                y_true=self.curr_true_b,
                y_guess='skip' if skip else self.curr_b,
        )

        write_header = not self.log_path.exists()

        with open(self.log_path, 'a') as f:
            csv = DictWriter(f, fieldnames=row.keys(), lineterminator='\n')
            if write_header:
                csv.writeheader()
            csv.writerow(row)

    def start_view_size_dialog(self):

        def set_view_size(x):
            self.view_size_A = float(x)
            self.render_curr_example()

        self.dialog_prompt = "View size (A):"
        self.dialog_callback = set_view_size

        cmd.refresh_wizard()

    def start_neighbor_distance_dialog(self):

        def set_neighbor_distance(x):
            self.neighbor_params.distance_A = float(x)
            self.load_frames()
            self.render_curr_example()

        self.dialog_prompt = "Neighbor distance (A):"
        self.dialog_callback = set_neighbor_distance

        cmd.refresh_wizard()

    def start_noise_max_distance_dialog(self):

        def set_noise_max_dist(x):
            self.neighbor_params.noise_max_distance_A = float(x)
            self.render_curr_example()

        self.dialog_prompt = "Noise max distance (A):"
        self.dialog_callback = set_noise_max_dist

        cmd.refresh_wizard()

    def start_noise_max_angle_dialog(self):

        def set_noise_max_angle(x):
            self.neighbor_params.noise_max_angle_deg = float(x)
            self.render_curr_example()

        self.dialog_prompt = "Noise max angle (deg):"
        self.dialog_callback = set_noise_max_angle

        cmd.refresh_wizard()

def mmgp_manual_classifier(*args, **kwargs):
    """
    DESCRIPTION

        Attempt to manually solve problems from the neighbor location dataset.

        The purpose of this wizard is to help give an intuitive sense for the 
        difficulty of different sets of neighbor location parameters.

    USAGE

        mmgp_manual_classifier [ db_path [, config_path [, log_path ]]]

    ARGUMENTS

        db_path = str: Path to the database file.

        config_path = str: Path to a file specifying various dataset parameters.
        This should be a NestedText file with the following fields:

            db_path: Database to use (overridden by above argument)
            db_split: Which split of the data to use; defaults to 'train'
            view_size_A: Length of the view in angstroms
            neighbor_distance_A: Distance between view centers in angstroms
            noise_max_distance_A: Maximum amount of noise to add to the distance
            noise_max_angle_deg: Maximum amount to rotate the neighboring image

        Any values specified by this file will override any values specified in 
        the following default configuration file, if it exists:

            {DEFAULT_CONFIG_PATH}
            
        log_path = str: Path to a file where the results of each manual 
        classification will be logged, in CSV format.  If specified, any 
        training examples that are present in the log file will be skipped.

    ENVIRONMENT VARIABLES

        PDB_MMCIF = str: Path to the directory containing mmCIF files for every 
        structure in the database, organized as in the PDB (e.g. `ab/1abc.cif`).

    HOTKEYS
        
        Tab: Cycle through possible positions for the second view
        Ctrl+Tab: Cycle in reverse

        Note that these hotkeys prevent the use of tab-completion while the 
        wizard is running.
            
    SEE ALSO

        mmgp_training_examples, voxelize, load_voxels
    """
    kwargs.pop('_self', None)
    wizard = ManualClassifier(*args, **kwargs)
    cmd.set_wizard(wizard)

mmgp_manual_classifier.__doc__ = mmgp_manual_classifier.__doc__.format(
        DEFAULT_CONFIG_PATH=ManualClassifier.DEFAULT_CONFIG_PATH,
)
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

