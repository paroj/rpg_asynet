FROM nvidia/cuda:11.2.2-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME "/usr/local/cuda"
ENV CUDA_VISIBLE_DEVICES 0

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC &&\
    apt-get update &&\
    apt-get install -y git g++=4:9.3.0-1ubuntu2 cmake=3.16.3-1ubuntu1 libgl1-mesa-glx ffmpeg libsm6=2:1.2.3-1 libxext6=2:1.3.4-0ubuntu1 wget

# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda_installer.sh &&\
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh -O miniconda_installer.sh &&\
    bash miniconda_installer.sh -b -p ./miniconda &&\
    eval "$(/miniconda/bin/conda shell.bash hook)" &&\
    conda init

ENV PATH="/miniconda/bin:$PATH"

# opencv
RUN conda install -y -c conda-forge tqdm=4.61.0 scikit-video=1.1.11 eigen=3.3.9 boost=1.74.0 boost-cpp=1.74.0 pybind11=2.6.2

WORKDIR /packages

# rpg_asynet setup
RUN git clone https://github.com/uzh-rpg/rpg_asynet.git
WORKDIR /packages/rpg_asynet
RUN git checkout b0262db06387ba93fb9abe6f0dd878dd3e76975f
#     apt-get install -y libfreetype6-dev &&\
#     ls &&\
#     ln -s /usr/include/freetype2/ /lib/X11/include
# install freetype2 so it can be found when installing matplotlib
# RUN wget https://download.savannah.gnu.org/releases/freetype/freetype-2.11.1.tar.gz -O freetype-2.11.1.tar.gz &&\
#     tar xzf freetype-2.11.1.tar.gz
# WORKDIR /packages/rpg_asynet/freetype-2.11.1
# RUN ls &&\
#     bash configure --prefix=/lib/X11 &&\
#     make &&\
#     make install
# WORKDIR /lib/X11/include
# RUN ls

# solves opencv-python install error
RUN apt-get update &&\
    apt-get install -y ninja-build

# run this first to avaoid unnecessary reinstalls, as this requirements.txt reuires exact versions
# sed command required for pc2815 and ampere_cluster
WORKDIR /packages
RUN conda install -y -c conda-forge matplotlib
# don't use 24s#.*#opencv-python==4.3.0.38#' anymore to 'fix' opencv-python issue 648
RUN sed -i '24s#.*#opencv-python==4.5.5.64#' rpg_asynet/requirements.txt &&\
    sed -i '22s#.*#numpy>=1.18.1#' rpg_asynet/requirements.txt &&\
    sed -i '47s#.*#torch==1.7.1#' rpg_asynet/requirements.txt &&\
    sed -i '48s#.*#torchvision==0.8.2#' rpg_asynet/requirements.txt &&\
    sed -i '19d' rpg_asynet/requirements.txt &&\
    sed -i '20d' rpg_asynet/requirements.txt &&\
    cat rpg_asynet/requirements.txt &&\
    pip install -r rpg_asynet/requirements.txt
# RUN sed -i '24s#.*#opencv-python==4.3.0.38#' rpg_asynet/requirements.txt &&\
#     sed -i '47s#.*#torch==1.7.1#' rpg_asynet/requirements.txt &&\
#     sed -i '48s#.*#torchvision==0.8.2#' rpg_asynet/requirements.txt &&\
#     sed -i '19d' rpg_asynet/requirements.txt &&\
#     cat rpg_asynet/requirements.txt &&\
#     pip install -r rpg_asynet/requirements.txt

# SparseConvNet setup
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN git clone https://github.com/facebookresearch/SparseConvNet.git &&\
    cd SparseConvNet &&\
    git checkout 64909718a47198669464ef7605ca937cc21b312d &&\
    cd ..
# ADD SparseConvNet/ /packages/SparseConvNet
# WORKDIR /packages/SparseConvNet
# RUN bash develop.sh

# event_representation_tool setup
WORKDIR /packages/rpg_asynet
RUN pip install event_representation_tool/

# async_sparse_py setup
WORKDIR /packages/rpg_asynet/async_sparse_py/include
# fetch libeigen and checkout last known compatible version
RUN git clone https://gitlab.com/libeigen/eigen.git
WORKDIR /packages/rpg_asynet/async_sparse_py/include/eigen/
RUN git checkout 05754100fecf00e13b2a5799e31570a980e4dd72
WORKDIR /packages/rpg_asynet/
RUN pip install async_sparse_py/

# rpg_vid2e setup
WORKDIR /packages/rpg_asynet/image_event_dataloader/
RUN git clone https://github.com/uzh-rpg/rpg_vid2e.git --recursive
WORKDIR /packages/rpg_asynet/image_event_dataloader/rpg_vid2e
RUN apt-get install -y libopencv-dev &&\
    pip install esim_py/

# pytorch_yolo dependencies
RUN pip install scikit-image>=0.14 matplotlib>=2.2.3 numpy>=1.15

# update pip because of 'numpy.core.multiarray failed to import' error
RUN pip install -U numpy

#COPY . .
#
#RUN pip install -r requirements.txt
#
#RUN cd SparseConvNet/ &&\
#    bash develop.sh
#
#RUN pip install event_representation_tool/
#
#RUN pip install async_sparse_py/
#
#RUN pip install image_event_dataloader/rpg_vid2e/esim_py/
#
#RUN pip install scikit-image>=0.14 matplotlib>=2.2.3 numpy>=1.15

# RUN apt-get update &&\
#     apt-get -y install gdb

# update setup file for SparseConvNet to force
# WORKDIR /packages
# # RUN pip uninstall --yes sparseconvnet
# RUN rm -f SparseConvNet/setup.py
# ADD SparseConvNet/setup.py /packages/SparseConvNet/setup.py
WORKDIR /code

CMD python -u /code/train_until_finished.py

# INSTALL SparseConvNet LIBRARY WITH GPU SUPPORT:
#
# after build: run container in interactive mode (./docker_debug.sh) and execute:
#
# cd /packages/SparseConvNet
# bash develop.sh
#
# then outside the container
#
# docker commit --change 'ENTRYPOINT ["bash", "entrypoint.sh"]' [container_id] rpg_asynet:latest
