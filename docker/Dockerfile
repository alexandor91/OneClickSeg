# 指定cuda镜像版本，详见https://hub.docker.com/r/nvidia/cuda/tags?page=1
ARG CUDA="11.6.1"
ARG CUDNN="8"
ARG UBUNTU="20.04"
FROM docker.io/nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu${UBUNTU}

LABEL maintainer "Zijian Yu"
MAINTAINER Zijian Yu "https://github.com/yzj2019"

# 环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    HTTPS_CERT=/etc/ssl/certs/ssl-cert-snakeoil.pem \
    HTTPS_CERT_KEY=/etc/ssl/private/ssl-cert-snakeoil.key \
    VGL_DISPLAY=egl \
    VNC_THREADS=2

# Copy config
COPY docker_config /docker_config
COPY resources /tmp/resources

# apt 换源
RUN sed -i s@/archive.ubuntu.com/@/mirrors.ustc.edu.cn/@g /etc/apt/sources.list && \
    apt-get clean && \
    apt-get update
# DNS服务器
# RUN echo "nameserver 8.8.8.8" >> /etc/resolv.conf
# RUN echo "nameserver 8.8.4.4" >> /etc/resolv.conf

# Install and Configure OpenGL
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libxau6 libxdmcp6 libxcb1 libxext6 libx11-6 \
        libglvnd0 libgl1 libglx0 libegl1 libgles2 \
        libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /usr/share/glvnd/egl_vendor.d/ && \
    echo "{\n\
\"file_format_version\" : \"1.0.0\",\n\
\"ICD\": {\n\
    \"library_path\": \"libEGL_nvidia.so.0\"\n\
}\n\
}" > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Install and Configure for Vulkan
RUN apt-get update && \
    apt-get install -y --no-install-recommends vulkan-tools &&\
    rm -rf /var/lib/apt/lists/* &&\
    VULKAN_API_VERSION=$(dpkg -s libvulkan1 | grep -oP 'Version: [0-9|\.]+' | grep -oP '[0-9]+(\.[0-9]+)(\.[0-9]+)') &&\
    mkdir -p /etc/vulkan/icd.d/ && \
    echo "{\n\
\"file_format_version\" : \"1.0.0\",\n\
\"ICD\": {\n\
    \"library_path\": \"libGLX_nvidia.so.0\",\n\
    \"api_version\" : \"${VULKAN_API_VERSION}\"\n\
}\n\
}" > /etc/vulkan/icd.d/nvidia_icd.json


# Install some common tools 
RUN apt-get update && \
    apt-get install -y apt-utils sudo vim gedit locales wget curl git gnupg2 lsb-release net-tools iputils-ping mesa-utils \
    openssh-server bash-completion software-properties-common python3-pip tmux \
    ninja-build cmake build-essential libopenblas-dev xterm xauth libopenexr-dev ssh &&\
    rm -rf /var/lib/apt/lists/*


# Install desktop
RUN apt-get update && \
    # add apt repo for firefox
    add-apt-repository -y ppa:mozillateam/ppa &&\
    mkdir -p /etc/apt/preferences.d &&\
    echo "Package: firefox*\n\
Pin: release o=LP-PPA-mozillateam\n\
Pin-Priority: 1001" > /etc/apt/preferences.d/mozilla-firefox &&\
    # install xfce4 and firefox
    apt-get install -y xfce4 terminator fonts-wqy-zenhei ffmpeg firefox dbus-x11 &&\
    # remove and disable screensaver
    apt-get remove -y xfce4-screensaver --purge &&\
    # set firefox as default web browser
    update-alternatives --set x-www-browser /usr/bin/firefox &&\
    rm -rf /var/lib/apt/lists/*

ENV DBUS_SYSTEM_BUS_ADDRESS=unix:path=/host/run/dbus/system_bus_socket
RUN apt-get update && apt-get install -y pulseaudio && mkdir -p /var/run/dbus &&\
    rm -rf /var/lib/apt/lists/*

# Install nomachine
RUN bash /docker_config/install_nomachine.sh

# Install miniconda && initialize shell
RUN bash /docker_config/install_miniconda.sh

# git config & ssh config
RUN git config --global user.email "1223358821@qq.com" &&\
    git config --global user.name "Zijian Yu" &&\
    # configuration ssh enviroment for github
    mkdir -p /root/.ssh &&\
    cp /docker_config/.ssh/id_rsa /root/.ssh &&\
    cp /docker_config/.ssh/config /root/.ssh &&\
    chmod -R 600 /root/.ssh &&\
    echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config &&\
    echo "UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config &&\
    # configuration ssh enviroment for login
    cp /docker_config/.ssh/id_rsa_docker.pub /root/.ssh &&\
    cat /root/.ssh/id_rsa_docker.pub >> /root/.ssh/authorized_keys &&\
    mkdir /var/run/sshd &&  \
    sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config && \
    sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

# 开放ssh和nomachine端口
EXPOSE 22 4000

# 设置启动
ENTRYPOINT ["/docker_config/entrypoint.sh"]