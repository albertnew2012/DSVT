FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV NVENCODE_CFLAGS="-I/usr/local/cuda/include"
ENV CV_VERSION=4.2.0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get clean && apt-get update && apt-get install -y sudo build-essential \
    software-properties-common build-essential git zip unzip libssl-dev lsb-release \
    libeigen3-dev curl cmake pkg-config python3-pip python3-tornado python3-dev \
    locales python3-cairo python3-gi python3-gi-cairo gir1.2-gtk-3.0
RUN sudo python3 -m pip install --upgrade pip

RUN locale-gen en_US.UTF-8

RUN apt install -y python3-opencv
RUN apt install -y vim
RUN apt install -y rsync
RUN apt install -y htop
RUN apt install -y nvtop

ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# Create your user
# RUN useradd -ms /bin/bash ubuntu
# RUN usermod -aG sudo ubuntu
# RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/ubuntu
# USER ubuntu

# # Set work dir as home
# ARG HOME=/home/ubuntu
# WORKDIR $HOME

# Set environment variables
# ENV HOME=$HOME \
#     NVIDIA_VISIBLE_DEVICES="all" \
#     NVIDIA_DRIVER_CAPABILITIES="video,compute,utility,graphics" \
#     LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib:/usr/lib:/usr/local/lib 
    # PATH=/home/ubuntu/.local/bin:$PATH

# RUN sudo ln -s /usr/bin/python3 /usr/bin/python

# # PyTorch for CUDA 11.6
# RUN pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
#     torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 \
#     torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.1+cu116.html

# # PyTorch for CUDA 11.8
# RUN pip3 install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
#     --index-url https://download.pytorch.org/whl/cu118 \
#     torch-scatter

# PyTorch for CUDA 11.8
# RUN pip3 install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
#     --index-url https://download.pytorch.org/whl/cu118 

# Install specific versions of torch, torchvision, and torchaudio
RUN pip3 install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# Install torch-scatter
# RUN pip3 install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
RUN pip3 install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

# # lidardet3d dependancies
# RUN pip3 install setuptools==69.5.1 wheel==0.43.0 numpy==1.23.0 \
#     llvmlite==0.41.1 numba==0.58.1 tensorboardX==2.6.2.2 easydict==1.13 pyyaml==6.0.1 \
#     scikit-image==0.21.0 tqdm==4.66.4 SharedArray==3.2.3 open3d==0.13.0 av2==0.2.1 \
#     kornia==0.6.5 pyquaternion==0.9.9 pypcd4==1.0.1 spconv-cu116==2.3.6 \
#     nuscenes-devkit==1.1.11 "motmetrics<=1.1.3" "pandas>=1.0.5" tabulate==0.9.0 \
#     plotly==5.22.0

# safest in a disposable image
RUN apt-get update && apt-get purge -y python3-blinker \
    && apt-get autoremove -y && apt-get clean

RUN pip3 install setuptools wheel numpy \
    llvmlite numba==0.58.1 tensorboardX easydict pyyaml \
    scikit-image tqdm SharedArray open3d av2 \
    kornia pyquaternion pypcd4 spconv-cu118 \
    nuscenes-devkit==1.1.11 tabulate\
    plotly gpustat

RUN pip3 install onnx
RUN pip3 install --extra-index-url https://pypi.nvidia.com tensorrt==8.6.0
RUN pip3 install onnxruntime-gpu


# USER root
# RUN apt install -y libcudnn8=8.6.0.163-1+cuda11.8
# RUN apt install -y libcudnn8-dev=8.6.0.163-1+cuda11.8
# RUN apt install -y libcudnn8-samples=8.9.2.26-1+cuda11.8

RUN apt install -y  \
    libnvinfer8=8.6.1.6-1+cuda11.8 \
    libnvinfer-plugin8=8.6.1.6-1+cuda11.8 \
    libnvinfer-vc-plugin8=8.6.1.6-1+cuda11.8 \
    libnvinfer-lean8=8.6.1.6-1+cuda11.8 \
    libnvinfer-dispatch8=8.6.1.6-1+cuda11.8 \
    libnvparsers8=8.6.1.6-1+cuda11.8 \
    libnvonnxparsers8=8.6.1.6-1+cuda11.8 \
    libnvinfer-bin=8.6.1.6-1+cuda11.8 \
    libnvinfer-dev=8.6.1.6-1+cuda11.8 \
    libnvinfer-lean-dev=8.6.1.6-1+cuda11.8 \
    libnvinfer-plugin-dev=8.6.1.6-1+cuda11.8 \
    libnvinfer-vc-plugin-dev=8.6.1.6-1+cuda11.8 \
    libnvinfer-dispatch-dev=8.6.1.6-1+cuda11.8 \
    libnvparsers-dev=8.6.1.6-1+cuda11.8 \
    libnvonnxparsers-dev=8.6.1.6-1+cuda11.8 \
    libnvinfer-samples=8.6.1.6-1+cuda11.8 \
    libnvinfer-headers-dev=8.6.1.6-1+cuda11.8 \
    libnvinfer-headers-plugin-dev=8.6.1.6-1+cuda11.8

RUN apt install -y  tensorrt=8.6.1.6-1+cuda11.8

ENV PATH="/usr/src/tensorrt/bin:${PATH}"



############################## install ros2 humble ################################
# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Add the ROS 2 GPG key
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
# Add the ROS 2 repository
RUN sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list'
# Update and install ROS Humble
RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*
# Source the ROS Humble setup script
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN rosdep init && rosdep update
RUN apt-get update &&  apt-get install -y ros-humble-diagnostics

# Install other dependencies or custom packages as needed
# RUN apt-get update && apt-get install -y <your-packages> && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV ROS_DISTRO=humble

############################## install ninja ################################
# Download Ninja using curl
RUN curl -L -o ninja-linux.zip https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux.zip
# Unzip the Ninja binary
RUN apt-get install -y unzip && unzip ninja-linux.zip
# Move Ninja to a directory in the PATH
RUN mv ninja /usr/local/bin/
# Verify installation (optional, useful for debugging)
RUN ninja --version


############################# install pcl ######################################
RUN apt-get update && apt-get install -y libpcl-dev && apt-get install -y python3-pcl
# RUN apt-get install -y python3-pcl

# RUN apt-get update && apt-get install -y python3-colcon-common-extensions && apt-get install -y gdb

# RUN git clone https://github.com/open-mmlab/OpenPCDet.git
# WORKDIR $HOME/OpenPCDet/
# RUN python3 setup.py develop --user

# Uninstall matplotlib
# otherwise it conflicts with cv 
# by defualt matplotlib is also installed with apt install
RUN pip3 uninstall -y matplotlib
RUN pip3 install tensorflow tensorboard


############################# install sys tools ######################################
RUN apt-get update && apt-get install -y libboost-filesystem-dev gdb x11-apps

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3.10-tk \
    python-is-python3 \
    libgomp1 \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libxext6 \
    libxrender1 \
    libxtst6 \
    x11-apps \
    net-tools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

####################################################################################################


# Dynamically create a user matching the host's username, UID, and GID
ARG USER_NAME=defaultuser
ARG USER_ID=1000
ARG GROUP_ID=1000

# Create the user, add to sudo group, and configure sudo privileges
RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -l -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USER_NAME} && \
    usermod -aG sudo ${USER_NAME} && \
    mkdir -p /etc/sudoers.d && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME} && \
    chmod 0440 /etc/sudoers.d/${USER_NAME} && \
    chmod 755 /etc/sudoers.d

# Fix permissions for apt directories
RUN chmod -R a+rwx /var/lib/apt/lists /var/cache/apt /var/log

# Verify sudo works before switching users
RUN echo "Testing sudo access" && \
    sudo -u ${USER_NAME} bash -c "echo Hello from user: $(whoami)"

# Switch to the dynamically created user
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}
