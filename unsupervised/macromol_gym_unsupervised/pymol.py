import pymol
import macromol_gym_unsupervised as mmgu
import macromol_dataframe as mmdf
import os

from pymol import cmd
from pymol.wizard import Wizard
from macromol_gym_unsupervised.database_io import open_db, select_zone_pdb_ids
from macromol_gym_unsupervised.torch import (
        MacromolImageDataset, InfiniteSampler,
)
from macromol_voxelize import Grid
from macromol_voxelize.pymol import (
        select_view, render_image, pick_channel_colors
)

class TrainingExamples(Wizard):

    def __init__(
            self,
            db_path,
            length_voxels=24,
            resolution_A=1,
            atom_radius_A=None,
            split='train',
    ):
        super().__init__()

        self.db = open_db(db_path)

        self.length_voxels = int(length_voxels)
        self.resolution_A = float(resolution_A)
        self.atom_radius_A = (
                float(atom_radius_A) if atom_radius_A else
                self.resolution_A / 2
        )
        self.channels = [['C'], ['N'], ['O'], ['*']]
        self.show_voxels = True
        self.scale_alpha = False

        self.dataset = MacromolImageDataset(
                db_path=db_path,
                split=split,
                img_params=self.make_image_params(),
        )
        sampler = InfiniteSampler(
                len(self.dataset),
                shuffle=True,
        )
        self.zone_order = list(sampler)
        self.i = 0
        self.random_seed = 0

        self.dialog_prompt = None
        self.dialog_callback = None
        self.dialog_input = ''

        self.redraw()

    def get_panel(self):
        panel = [
                [1, "Neighbor Dataset", ''],
                [2, "Next <C-Space>", 'cmd.get_wizard().next_training_example()'],
                [2, "Previous", 'cmd.get_wizard().prev_training_example()'],
                [2, "New random seed", 'cmd.get_wizard().new_random_seed()'],
                [2, f"Image length: {self.length_voxels} voxels", 'cmd.get_wizard().start_image_length_dialog()'],
                [2, f"Image resolution: {self.resolution_A}A", 'cmd.get_wizard().start_image_resolution_dialog()'],
                [2, f"Atom radius: {self.atom_radius_A}A", 'cmd.get_wizard().start_atom_radius_dialog()'],
                [3, f"Show voxels: {'yes' if self.show_voxels else 'no'}", 'show_voxels'],
                [3, f"Scale alpha: {'yes' if self.scale_alpha else 'no'}", 'scale_alpha'],
                [2, "Done", 'cmd.set_wizard()'],
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
        if self.dialog_prompt is not None:
            return [f"{self.dialog_prompt} \\999{self.dialog_input}"]
        else:
            return [f"Zone: {self.curr_zone_id}"]

    def do_key(self, key, x, y, mod):
        # See `wt_vs_mut` for details on how this method works.

        BACKSPACE = (8, 27)
        ENTER = (10, 13)
        CTRL_SPACE = (0, 2)

        if self.dialog_prompt is not None:
            print(key)
            if key in BACKSPACE:
                self.dialog_input = self.dialog_input[:-1]
            elif key >= 32:
                self.dialog_input += chr(key)
            elif key in ENTER:
                self.dialog_callback(self.dialog_input)
                self.dialog_prompt = None
                self.dialog_callback = None
                self.dialog_input = ''
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
            self.length_voxels = int(x)
            self.update_image_params()

        self.dialog_prompt = "Image length (voxels):"
        self.dialog_callback = set_image_length

        cmd.refresh_wizard()

    def start_image_resolution_dialog(self):

        def set_image_resolution(x):
            self.resolution_A = float(x)
            self.update_image_params()

        self.dialog_prompt = "Image resolution (A):"
        self.dialog_callback = set_image_resolution

        cmd.refresh_wizard()

    def start_atom_radius_dialog(self):

        def set_atom_radius(x):
            self.atom_radius_A = float(x)
            self.update_image_params()

        self.dialog_prompt = "Atom radius (A):"
        self.dialog_callback = set_atom_radius

        cmd.refresh_wizard()

    def make_image_params(self):
        return mmgu.ImageParams(
                grid=Grid(
                    length_voxels=self.length_voxels,
                    resolution_A=self.resolution_A,
                ),
                atom_radius_A=self.atom_radius_A,
                element_channels=self.channels,
        )

    def update_image_params(self):
        self.dataset.img_params = self.make_image_params()

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
        i = self.zone_order[self.i] + self.random_seed * len(self.dataset)
        x = self.dataset[i]

        self.curr_zone_id = x['zone_id']

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

        # Render the image
        select_view(
                name='atoms',
                sele=curr_pdb_obj,
                grid=self.dataset.img_params.grid,
                frame_ix=x['frame_ia'],
        )
        if self.show_voxels:
            render_image(
                    img=x['image'].detach().numpy(),
                    grid=self.dataset.img_params.grid,
                    frame_xi=mmdf.invert_coord_frame(x['frame_ia']),
                    obj_names=dict(
                        voxels='voxels',
                        outline='outline',
                    ),
                    channel_colors=pick_channel_colors('atoms', self.channels),
                    outline=(1, 1, 0),
                    scale_alpha=self.scale_alpha,
            )
            cmd.show('sticks', 'byres atoms')

        if not keep_view:
            cmd.zoom('atoms', buffer=10)
            cmd.center('atoms')

def mmgu_training_examples(db_path, *args, **kwargs):
    kwargs.pop('_self', None)  # Don't know why this argument exists...
    wizard = TrainingExamples(db_path, *args, **kwargs)
    cmd.set_wizard(wizard)

pymol.cmd.extend('mmgu_training_examples', mmgu_training_examples)

