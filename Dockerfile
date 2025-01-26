FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        curl \
        libglib2.0-0 \
        libvulkan1 \
        vulkan-tools \
        libglvnd0 \
        libglu-dev \
        libegl1 \
        libxext6 \
        libx11-6 \
        python3-rospy \
        libboost-math-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"
RUN sh /uv-installer.sh && rm /uv-installer.sh && \
    uv python install 3.10.13

# to use SimplerEnv (depends on Vulkan, with NVIDIA GPU)
COPY .devcontainer/nvidia_icd.json /usr/share/vulkan/icd.d/

# Install project specific dependencies
WORKDIR /root/workspace
RUN git clone https://github.com/allenzren/SimplerEnv --recurse-submodules
COPY pyproject.toml /root/workspace/
RUN uv sync && \
    uv pip install -e . ./SimplerEnv --no-config && \
    uv pip install -e . ./SimplerEnv/ManiSkill2_real2sim --no-config && \
    rm pyproject.toml && \
    uv pip install --extra-index-url https://rospypi.github.io/simple rospy-all tf tf2_ros
#   uv pip install --extra-index-url https://rospypi.github.io/simple cv_bridge
#   uv pip install /root/workspace/open-pi-zero

RUN echo '[ -z "$VIRTUAL_ENV" ] && source /root/workspace/.venv/bin/activate' >> /root/.bashrc
WORKDIR /root/workspace/open-pi-zero
CMD ["/bin/bash"]
