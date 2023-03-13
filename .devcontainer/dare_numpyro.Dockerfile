# syntax=docker/dockerfile:1
# load a base image
FROM ubuntu:18.04

# create some directories to mount local data (volume)
RUN mkdir /project

# Install ubuntu libraries and packages
RUN apt-get update -y && \
    apt-get install git curl build-essential vim -y
# install python
ENV PATH="/build/miniconda3/bin:${PATH}"
ARG PATH="/build/miniconda3/bin:${PATH}"
RUN mkdir /build && \
    mkdir /build/.conda
#Install Python3.9 via miniconda
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /build/miniconda3 &&\
    rm -rf /Miniconda3-latest-Linux-x86_64.sh
WORKDIR /build

# create a python environment
RUN conda install python=3.9
RUN conda install numpy pandas scikit-learn seaborn ipywidgets openpyxl
RUN conda install jupyterlab
RUN pip install --upgrade pip
RUN pip install jax jaxlib
RUN pip install optax flax
RUN pip install graphviz clu
RUN pip install numpyro
RUN conda install -c conda-forge arviz
