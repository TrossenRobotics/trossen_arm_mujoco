# Trossen Arm MuJoCo

Create Conda Enviroment with Python 3.13 or above

```bash
pip install dm_control
```

```bash
python3 scripts/sim_env.py
```

# Set the environment variable (This is usefull if you run into Mesa Loader Issue)

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

---

# **Aloha Simulation Package**

## **Overview**  
The **Aloha Simulation Package** provides the necessary assets and scripts for simulating and training robotic policies using the **Aloha Solo** system in **MuJoCo**. It includes **URDFs, mesh models, and MuJoCo XML files** for the robot configuration, as well as Python scripts for policy execution, reward-based evaluation, data collection, and visualization.

This package supports two types of simulation environments:  
1. **End-Effector (EE) Controlled Simulation (`ee_sim_env.py`)** – Uses **motion capture (mocap) bodies** to move the arms.  
2. **Joint-Controlled Simulation (`sim_env.py`)** – Uses **position controllers** for more realistic robot movements.  

---

## **Folder Structure**  

```
📦 Aloha Solo Simulation Package  
 ┣ 📂 assets/  
 ┃ ┣ 📜 MuJoCo XML files  
 ┃ ┣ 📜 URDF files  
 ┃ ┗ 📂 meshes/  → 3D model files for simulation  
 ┣ 📂 scripts/  
 ┃ ┣ 📜 constants.py  
 ┃ ┣ 📜 ee_sim_env.py  
 ┃ ┣ 📜 sim_env.py  
 ┃ ┣ 📜 scripted_policy.py  
 ┃ ┣ 📜 record_sim_episodes.py  
 ┃ ┣ 📜 visualize.py  
 ┃ ┗ 📜 utils.py  
 ┗ 📜 README.md  
```

---

## **1. Assets Folder (`assets/`)**  

This folder contains all required **MuJoCo XML configuration files**, **URDF files**, and **mesh models** for the simulation.  

### **Key Files:**
- **`aloha.xml`** → Base model definition of the Aloha Solo robot.  
- **`aloha_scene.xml`** → Uses **motion capture (mocap) bodies** to control the arms in simulation.  
- **`aloha_scene_joint.xml`** → Uses **joint controllers** similar to real hardware, enabling precise control over movements.  
- **`wxai_follower.urdf` & `wxai_follower.xml`** → URDF and XML descriptions of the follower arms.  
- **`meshes/`** → Contains **STL** and **OBJ** files for the robot components, including arms, cameras, and environmental objects.

### **Motion Capture vs Joint-Controlled Environments:**
- **Motion Capture (`aloha_scene.xml`)**: Uses predefined **mocap bodies** that move the robot arms dynamically based on scripted policies.  
- **Joint Control (`aloha_scene_joint.xml`)**: Uses position controllers for each joint, similar to a real-world robot setup.

---

## **2. Scripts Folder (`scripts/`)**  

This folder contains all Python scripts necessary for **running simulations, executing policies, recording episodes, and visualizing results**.

### **2.1 Simulation Scripts**
- **`ee_sim_env.py`**  
  - Loads **`aloha_scene.xml`** (motion capture-based control).  
  - The arms move by following the positions commanded to the **mocap bodies**.  
  - Used for generating **scripted policies** that control the robot’s arms in predefined ways.

- **`sim_env.py`**  
  - Loads **`aloha_scene_joint.xml`** (position-controlled joints).  
  - Uses **joint controllers** instead of mocap bodies.  
  - Mimics the real robot’s movement with controlled joint actuation.

### **2.2 Scripted Policy Execution**
- **`scripted_policy.py`**  
  - Defines **pre-scripted movements** for the robot arms to perform tasks like picking up objects.  
  - Uses the **motion capture bodies** to generate smooth movement trajectories.  
  - In the current setup, a policy is designed to **pick up a red block**, with **randomized block positions** in the environment.
I've refined the explanation in the **README.md** to ensure clarity and completeness. Here's the updated section:

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
   - Each episode is saved as **`episode_X.hdf5`** inside the **`aloha_data/`** folder.  

4. **Visualizing the Data**  
   - The stored **HDF5 files** can be converted into videos using **`visualize.py`**.  
   - The resulting videos are saved in **MP4 format** inside the **`ee_sim_episodes_output/`** folder.  


---
