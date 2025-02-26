#!/bin/bash

# Update system packages
sudo apt update

# Install necessary dependencies
sudo apt install --reinstall libgl1-mesa-glx libgl1-mesa-dri -y
sudo apt install mesa-utils -y
sudo apt install libglfw3 libglfw3-dev -y

# Export environment variable
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
source ~/.bashrc

echo "Setup completed. Please restart your terminal or run 'source ~/.bashrc' for changes to take effect."
