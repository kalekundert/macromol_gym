import pymol
import macromol_voxelize as mmvox
import macromol_dataframe as mmdf
import numpy as np
import nestedtext as nt
import os

from pymol import cmd
from pymol.wizard import Wizard
from torch_deterministic import InfiniteSampler
from macromol_gym_unsupervised.torch import (
        MacromolDataset, ImageParams,
        open_db, select_zone_pdb_ids, make_unsupervised_image_sample,
)
from macromol_voxelize import Grid
from macromol_voxelize.pymol import (
        select_view, render_image, pick_channel_colors
)
from pathlib import Path
from functools import partial
from collections import ChainMap
from platformdirs import user_config_path
from voluptuous import Schema, Coerce

IMG_PARAMS_DEFAULT_PATH = user_config_path('macromol_gym') / 'img_params.nt'

class TrainingExamples(Wizard):
    DEFAULT_CONFIG_PATH = user_config_path('macromol_gym') / 'mmgu_training_examples.nt'

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
                [1, "Unsupervised Dataset", ''],
                [2, "Next <C-Space>", 'cmd.get_wizard().next_training_example()'],
                [2, "Previous", 'cmd.get_wizard().prev_training_example()'],
                [2, "New random seed", 'cmd.get_wizard().new_random_seed()'],
                [2, f"Save image: {self.curr_out_path}", 'cmd.get_wizard().save_image()'],
                [2, "Done", 'cmd.set_wizard()'],

                [1, "\\555Parameters", ''],
                [2, f"Image length: \\090{self.img_params.grid.length_voxels} voxels", 'cmd.get_wizard().start_image_length_dialog()'],
                [2, f"Image resolution: \\090{self.img_params.grid.resolution_A}A", 'cmd.get_wizard().start_image_resolution_dialog()'],
                [2, f"Atom radius: \\090{self.img_params.atom_radius_A}A", 'cmd.get_wizard().start_atom_radius_dialog()'],
                [3, f"Show voxels: {'yes' if self.show_voxels else 'no'}", 'show_voxels'],
                [3, f"Scale alpha: {'yes' if self.scale_alpha else 'no'}", 'scale_alpha'],
        ]
        return panel

    def get_menu(self, tag):
        menus = {
                'length_voxels': [[2, 'Image length', '']],
                'resolution_A': [[2, 'Image resolution', '']],
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
            'length_voxels': Coerce(int),
            'resolution_A': Coerce(float),
            'atom_radius_A': Coerce(float),
            'element_channels': [[str]],
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
            'length_voxels': 24,
            'resolution_A': 1.0,
            'element_channels': [['C'], ['N'], ['O'], ['*']],
            'show_voxels': True,
            'scale_alpha': False,
        })

        self.db_path = Path(config_chain['db_path'])
        self.db_split = config_chain['db_split']

        grid = Grid(
                length_voxels=config_chain['length_voxels'],
                resolution_A=config_chain['resolution_A'],
        )
        self.img_params = ImageParams(
            grid=grid,
            atom_radius_A=config_chain.get('atom_radius_A', grid.resolution_A / 2),
            element_channels=config_chain['element_channels'],
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
                    make_unsupervised_image_sample,
                    img_params=self.img_params,
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
            self.img_params.grid = mmvox.Grid(
                    length_voxels=int(x),
                    resolution_A=self.img_params.grid.resolution_A,
            )

        self.dialog_prompt = "Image length (voxels):"
        self.dialog_callback = set_image_length

        cmd.refresh_wizard()

    def start_image_resolution_dialog(self):

        def set_image_resolution(x):
            self.img_params.grid = mmvox.Grid(
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

    def set_show_voxels(self, value):
        self.show_voxels = value
        self.redraw()

    def set_scale_alpha(self, value):
        self.scale_alpha = value
        self.redraw()

    def save_image(self):
        np.save(self.curr_out_path, self.curr_image)
        print(f"Image data save to: {self.curr_out_path}")

    @property
    def curr_out_path(self):
        return f'{self.db_path.stem}_{self.curr_zone_id}.npy'

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
        self.curr_image = x['image']

        # Load the relevant structure:
        zone_pdb = select_zone_pdb_ids(self.db, x['zone_id'])
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

        # Render the image:
        select_view(
                name='img_atoms',
                sele=curr_pdb_obj,
                grid=self.img_params.grid,
                frame_ix=x['frame_ia'],
        )
        render_image(
                img=self.curr_image if self.show_voxels else None,
                grid=self.img_params.grid,
                frame_xi=mmdf.invert_coord_frame(x['frame_ia']),
                obj_names=dict(
                    voxels='voxels',
                    outline='outline',
                ),
                channel_colors=pick_channel_colors('img_atoms', self.img_params.element_channels),
                outline=(1, 1, 0),
                scale_alpha=self.scale_alpha,
        )

        if self.show_voxels:
            cmd.show('sticks', 'byres img_atoms')

        if not keep_view:
            cmd.zoom('img_atoms', buffer=10)
            cmd.center('img_atoms')

def mmgu_training_examples(*args, **kwargs):
    """
    DESCRIPTION
    
        Visualize unsupervised training samples.

    USAGE

        mmgu_training_examples db_path [, img_params_path [, split]] 

    ARGUMENTS

        db_path = str: Path to the database file.

        config_path = str: Path to a file specifying various dataset 
        parameters.  This should be a NestedText file with the following keys:
            
            db_path: Database to use (overridden by above argument)
            db_split: Which split of the data to use; defaults to 'train'
            length_voxels: Length of the image in voxels
            resolution_A: Length of each voxel in angstroms
            atom_radius_A: Radius of each atom in angstroms
            element_channels: List of lists of element symbols
            show_voxels: Whether to show the images; 'yes' or 'no'
            scale_alpha: Whether to scale the alpha channel; 'yes' or 'no'

        Any values specified by this file will override any values specified in 
        the following default image parameters file, if it exists:

            {DEFAULT_CONFIG_PATH}

    ENVIRONMENT VARIABLES

        PDB_MMCIF = str: Path to the directory containing mmCIF files for every 
        structure in the database, organized as in the PDB (e.g. `ab/1abc.cif`).

    HOTKEYS:

        Ctrl+Space: Advance to the next training example.

    SEE ALSO

        mmgp_training_examples, voxelize, load_voxels
    """
    kwargs.pop('_self', None)  # Don't know why this argument exists...
    wizard = TrainingExamples(*args, **kwargs)
    cmd.set_wizard(wizard)

mmgu_training_examples.__doc__ = mmgu_training_examples.__doc__.format(
        DEFAULT_CONFIG_PATH=TrainingExamples.DEFAULT_CONFIG_PATH,
)
pymol.cmd.extend('mmgu_training_examples', mmgu_training_examples)

