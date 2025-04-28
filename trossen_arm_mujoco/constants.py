# Copyright 2025 Trossen Robotics
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from importlib.resources import files
import os
import numpy as np
import getpass

# Placeholder for the user name
USER_NAME = "trossen_ai"

ROOT_DIR = os.path.expanduser("~/.trossen/mujoco/")
DATA_DIR = "data/sim_transfer_cube"

### Simulated task configurations

SIM_TASK_CONFIGS = {
    "sim_transfer_cube": {
        "root_dir": ROOT_DIR,
        "data_dir": DATA_DIR,
        "num_episodes": 1,
        "episode_len": 600,
        "onscreen_render": False,
        "inject_noise": False,
        "cam_names": ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"],
    }
}

### Simulation envs fixed constants
DT = 0.02
START_ARM_POSE = [
    0.0, np.pi/12, np.pi/12, 0.0, 0.0, 0.0, 0.044, 0.044,
    0.0, np.pi/12, np.pi/12, 0.0, 0.0, 0.0, 0.044, 0.044
]

# Get the path to the assets directory
ASSETS_DIR = str(files("trossen_arm_mujoco").joinpath("assets"))

BOX_POSE = [None]
