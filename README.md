# Trossen Arm MujoCo

## **Overview**  

The **Trossen Arm MujoCo** provides the necessary assets and scripts for simulating and training robotic policies using the **Trossen AI** system in **MujoCo**. 
It includes **URDFs, mesh models, and MujoCo XML files** for the robot configuration, as well as Python scripts for policy execution, reward-based evaluation, data collection, and visualization.

This package supports two types of simulation environments:  
1. **End-Effector (EE) Controlled Simulation (`ee_sim_env.py`)** – Uses **motion capture (mocap) bodies** to move the arms.  
2. **Joint-Controlled Simulation (`sim_env.py`)** – Uses **position controllers** for more realistic robot movements.  

---

## Installation

First, clone this repository to your preferred directory:

It is recommended to create a virtual environment before installing dependencies. 
Create a Conda enviroment with Python 3.10 or above.

```bash
conda create --name mujoco_env python=3.10
```

After creation, activate the environment with:
```bash
conda activate mujoco_env
```

Install dependencies using:
```bash
pip install -r requirements.txt
```

To verify the installation, run:
```bash
python ee_sim_env.py
```
If the simulation window appears, the setup is successful.

### Set the environment variable (This is useful if you run into Mesa Loader Issues)

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

---

## **Folder Structure**  

```
Trossen Arm MujoCo
 ┣ assets/  
 ┃ ┣ MujoCo XML files  
 ┃ ┣ URDF files  
 ┃ ┗ meshes/  → 3D model files for simulation  
 ┣ scripts/  
 ┃ ┣ constants.py  
 ┃ ┣ ee_sim_env.py  
 ┃ ┣ sim_env.py  
 ┃ ┣ scripted_policy.py  
 ┃ ┣ record_sim_episodes.py  
 ┃ ┣ visualize.py  
 ┃ ┗ utils.py  
 ┣ requirements.txt  
 ┗ README.md

```

## **1. Assets Folder (`assets/`)**  

This folder contains all required **MujoCo XML configuration files**, **URDF files**, and **mesh models** for the simulation.  

### **Key Files:**

- **`trossen_ai.xml`** → Base model definition of the Trossen AI robot.  
- **`trossen_ai_scene.xml`** → Uses **motion capture (mocap) bodies** to control the arms in simulation.  
- **`trossen_ai_scene_joint.xml`** → Uses **joint controllers** similar to real hardware, enabling precise control over movements.  
- **`wxai_follower.urdf` & `wxai_follower.xml`** → URDF and XML descriptions of the follower arms.  
- **`meshes/`** → Contains **STL** and **OBJ** files for the robot components, including arms, cameras, and environmental objects.

### **Motion Capture vs Joint-Controlled Environments:**

- **Motion Capture (`trossen_ai_scene.xml`)**: Uses predefined **mocap bodies** that move the robot arms dynamically based on scripted policies.  
- **Joint Control (`trossen_ai_scene_joint.xml`)**: Uses position controllers for each joint, similar to a real-world robot setup.

---

## **2. Scripts Folder (`scripts/`)**  

This folder contains all Python scripts necessary for **running simulations, executing policies, recording episodes, and visualizing results**.

### **2.1 Simulation Scripts**

- **`ee_sim_env.py`**  
  - Loads **`trossen_ai_scene.xml`** (motion capture-based control).  
  - The arms move by following the positions commanded to the **mocap bodies**.  
  - Used for generating **scripted policies** that control the robot’s arms in predefined ways.

- **`sim_env.py`**  
  - Loads **`trossen_ai_scene_joint.xml`** (position-controlled joints).  
  - Uses **joint controllers** instead of mocap bodies.  
  - Mimics the real robot’s movement with controlled joint actuation.

### **2.2 Scripted Policy Execution**

- **`scripted_policy.py`**  
  - Defines **pre-scripted movements** for the robot arms to perform tasks like picking up objects.  
  - Uses the **motion capture bodies** to generate smooth movement trajectories.  
  - In the current setup, a policy is designed to **pick up a red block**, with **randomized block positions** in the environment.

---

## **3. How the Data Collection Works**  

The data collection process involves **two simulation phases**:  
1. **Running a scripted policy in `ee_sim_env.py` to record observations (joint positions).**  
2. **Replaying the recorded joint positions in `sim_env.py` to capture full episode data.**  

### **Step-by-Step Process**  

1. **Run `record_sim_episodes.py`**  
   - Starts **`ee_sim_env.py`** and executes a **scripted policy**.  
   - Captures **observations** in the form of **joint positions**.  
   - Saves these joint positions for later replay.

2. **Replay in `sim_env.py`**  
   - Uses the previously recorded **joint positions** as **input commands**.  
   - Replays the episode in the joint-controlled simulation (`sim_env.py`).  
   - Captures additional data, including:  
     - **Camera feeds from 4 different viewpoints**  
     - **Joint states** (actual positions during execution)  
     - **Actions** (input joint positions)  
     - **Reward values** to determine success or failure  

3. **Save the Data**  
   - All **observations and actions** are stored in **HDF5 format**, with one file per episode.  
   - Each episode is saved as **`episode_X.hdf5`** inside the **`trossen_ai_data/`** folder.  

4. **Visualizing the Data**  
   - The stored **HDF5 files** can be converted into videos using **`visualize.py`**.  
   - The resulting videos are saved in **MP4 format** inside the **`ee_sim_episodes_output/`** folder.  

---

## **4. Function Arguments Explanation**

### **a. record_sim_episodes.py**

To generate and save simulation episodes, use:
```bash
python record_sim_episodes.py \
    --task_name sim_transfer_cube \
    --num_episodes 5 \
    --dataset_dir trossen_ai_data/episodes \
    --onscreen_render
```
**Arguments:**

- `--task_name`: Name of the task (default: sim_transfer_cube)

- `--num_episodes`: Number of episodes to generate

- `--dataset_dir`: Directory where episodes will be saved

- `--episode_len`: Length of each episode (default: 400 steps)

- `--onscreen_render` : Enable real-time visualization (optional)

- `--inject_noise`: Add noise to actions for variation (optional)

- `--camera_names`: Comma-separated list of camera names for image collection

### **b. scripted_policy.py**

A predefined scripted policy can be used to control the robot:
```bash
python scripted_policy.py \
    --task_name sim_transfer_cube \
    --num_episodes 2
```
**Arguments:**

- `--task_name`: Name of the task

- `--num_episodes`: Number of episodes to run

- `--episode_len`: Length of each episode

- `--onscreen_render`: Enable visualization

- `--inject_noise`: Add noise to actions

### **c. visualize.py**

To convert saved episodes to videos, run:
```bash
python visualize.py \
    --dataset_dir data/episodes \
    --output_dir data/videos --fps 50
```
**Arguments:**

- `--dataset_dir`: Directory containing `.hdf5` files

- `--output_dir`: Output directory for `.mp4` files

- `--fps`: Frames per second for the generated videos (default: 50)

---

## **Customization**

### **1. Modifying Tasks**

To create a custom task, modify `ee_sim_env.py` or `sim_env.py` and define a new subclass of `BimanualViperXTask`. Implement:
- `initialize_episode(self, physics)`: Set up the initial environment state, including robot and object positions.
- `get_observation(self, physics)`: Define what sensor data and environment variables should be recorded as observations.
- `get_reward(self, physics)`: Implement the reward function to determine task success criteria.

### **2. Changing Policy Behavior**

Modify `scripted_policy.py` to define new behavior for the robotic arms. 
Update the trajectory generation logic in `PickAndTransferPolicy.generate_trajectory()` to create different movement patterns.

Each movement step in the trajectory is defined by:
- `t`: The time step at which the movement shall occur.
- `xyz`: The target position of the end effector in 3D space.
- `quat`: The target orientation of the end effector, represented as a quaternion.
- `gripper`: The target gripper opening width 0~1 where 0 is closed and 1 is fully open.

Example:
```bash
def generate_trajectory(self, ts_first):
    self.left_trajectory = [
        {"t": 0, "xyz": [0, 0, 0.4], "quat": [1, 0, 0, 0], "gripper": 0},
        {"t": 100, "xyz": [0.1, 0, 0.3], "quat": [1, 0, 0, 0], "gripper": 0.05}
    ]
```

### **3. Adding New Environment Setups**

The simulation uses XML files stored in the `assets/` directory. To introduce a new environment setup:

1. Create a new XML configuration file in `assets/` with desired object placements and constraints.

2. Modify `sim_env.py` to load the new environment by specifying the new XML file.

3. Update the scripted policies in `scripted_policy.py` to accommodate new task goals and constraints.

### **4. Adding New Sensors or Observations**

To record additional observations, modify `get_observation(self, physics)` in `sim_env.py`. Example:
```bash
def get_observation(self, physics):
    obs = super().get_observation(physics)
    obs["force_sensor"] = physics.data.sensordata.copy()  # Record force sensor data
    return obs
```

## Troubleshoot

if you encounter rendering issues or need a clean MujoCo setup on Linux.
```bash
chmod +x setup.sh
./setup.sh
```
Here's what it does:
- Reinstalls OpenGL-related libraries (`libgl1-mesa-glx`, `libgl1-mesa-dri`, `mesa-utils`) to fix rendering issues.
- Installs GLFW (`libglfw3`, `libglfw3-dev`), required for MujoCo simulations.
- Sets the `MUJOCO_GL=egl` environment variable to enable headless rendering using EGL.
- Updates the `.bashrc` file so the environment variable persists across terminal sessions.
- Prompts the user to restart the terminal or reload `.bashrc` for changes to take effect.