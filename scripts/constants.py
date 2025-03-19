# flake8: noqa

import os
import pathlib

DATA_DIR = os.path.expanduser('~/trossen_ai_data')

### Simulated task configurations

SIM_TASK_CONFIGS = {
    'sim_transfer_cube':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube',
        'num_episodes': 3,
        'episode_len': 400,
        'onscreen_render': True,
        'inject_noise': False,
        'camera_names': ["camera_high", "camera_low", "camera_left_wrist", "camera_right_wrist"]
    },

    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 3,
        'episode_len': 400,
        'onscreen_render': True,
        'inject_noise': False,
        'camera_names': ["camera_high", "camera_low", "camera_left_wrist", "camera_right_wrist"]
    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 3,
        'episode_len': 400,
        'onscreen_render': True,
        'inject_noise': False,
        'camera_names': ["camera_high", "camera_low", "camera_left_wrist", "camera_right_wrist"]
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 3,
        'episode_len': 400,
        'onscreen_render': True,
        'inject_noise': False,
        'camera_names': ["camera_high", "camera_low", "camera_left_wrist", "camera_right_wrist"]
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 3,
        'episode_len': 400,
        'onscreen_render': True,
        'inject_noise': False,
        'camera_names': ["camera_high", "camera_low", "camera_left_wrist", "camera_right_wrist"]
    },
}

### Simulation envs fixed constants
DT = 0.02
START_ARM_POSE = [
    0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.02239, -0.02239,
    0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.02239, -0.02239,
]
XML_DIR = str(pathlib.Path(__file__).parent.parent.resolve()) + '/assets/' # note: absolute path
BOX_POSE = [None]
