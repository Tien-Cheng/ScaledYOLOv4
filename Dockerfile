# docker build -t scaled_yolov4 --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .

FROM nvcr.io/nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV cwd="/home/"
WORKDIR $cwd

RUN apt-get -y update
# RUN apt-get -y upgrade

RUN apt-get install -y \
    software-properties-common \
    build-essential \
    checkinstall \
    cmake \
    pkg-config \
    yasm \
    git \
    vim \
    curl \
    wget \
    gfortran \
    libjpeg8-dev \
    libpng-dev \
    libtiff5-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libdc1394-22-dev \
    libxine2-dev \
    sudo \
    apt-transport-https \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    dbus-x11 \
    vlc \
    iputils-ping \
    python3-dev \
    python3-pip

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata python3-tk

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove

RUN rm -rf /var/cache/apt/archives/

### APT END ###

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN python3 -m pip install --no-cache-dir --upgrade pip

# COPY requirements.txt requirements.txt
# RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install --no-cache-dir \
    numpy==1.19.4 \
    tqdm==4.54.1 \
    pyyaml==5.3.1 \
    matplotlib==3.3.3 \
    tensorboard==2.4.0

RUN pip3 install --no-cache-dir  \
    opencv-python==4.4.0.46 \
    Pillow==8.0.1 \
    scipy==1.5.4

RUN pip3 install --no-cache-dir \
    torch==1.7.0 \
    torchvision==0.8.1

# install mish_cuda
RUN cd / && git clone https://github.com/JunnYu/mish-cuda && cd mish-cuda && python3 setup.py build install

# minimal Dockerfile which expects to receive build-time arguments, and creates a new user called “user” (put at end of Dockerfile)
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user