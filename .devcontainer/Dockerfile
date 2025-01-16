FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        curl \
        libvulkan1 \
        vulkan-tools \
        libgl1 \
        libglib2.0-0 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"
RUN uv python install 3.10.13

#RUN mkdir /root/workspace
#WORKDIR /root/workspace
#RUN git clone https://github.com/allenzren/SimplerEnv --recurse-submodules && \
#    git clone https://github.com/allenzren/open-pi-zero
#WORKDIR /root/workspace/open-pi-zero
#RUN uv sync && \
#    uv pip install -e . ../SimplerEnv && \
#    uv pip install -e . ../SimplerEnv/ManiSkill2_real2sim

WORKDIR /root/workspace/open-pi-zero

CMD ["/bin/bash"]`
