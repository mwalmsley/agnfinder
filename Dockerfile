FROM tensorflow/tensorflow:nightly-gpu-py3
# FROM nvcr.io/nvidia/tensorrt:20.02-py3

# # assumes fsps and agnfinder packages are in the docker context
# # get latest package list
# RUN apt-get update

# RUN apt-get update && apt-get install -y \
#     gcc \
#     gfortran \
#     make \
#     git 

# ## install base system requirements
# RUN apt-get --assume-yes install gcc
# RUN apt-get --assume-yes install gfortran
# RUN apt-get --assume-yes install make
# RUN apt-get --assume-yes install git

# RUN cd fsps/src && make
# ENV SPS_HOME=$HOME/fsps

# install Python via miniconda
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh
# install in silent mode, prompts are awkward
# RUN bash $HOME/miniconda.sh -b -p $HOME/miniconda

# ENV CATALOG_LOC=agnfinder/data/cpz_paper_sample_week3.parquet

# RUN eval "$(/root/miniconda/bin/conda shell.bash hook)" && conda init
# RUN eval "$(/root/miniconda/bin/conda shell.bash hook)" && conda create --name agnfinder numpy scipy
# RUN eval "$(/root/miniconda/bin/conda shell.bash hook)" && conda activate agnfinder && cd $agnfinder && pip install .
# RUN echo "conda activate agnfinder" >> ~/.bash_profile
# RUN echo "conda activate agnfinder" >> ~/.bashrc

# RUN pip install -r agnfinder/requirements.txt

# RUN mv -r filters $HOME/miniconda/envs/agnfinder/lib/python3.7/site-packages/sedpy
# see https://docs.docker.com/develop/develop-images/dockerfile_best-practices/

ADD agnfinder agnfinder/agnfinder
ADD results agnfinder/results

WORKDIR /agnfinder
CMD python3 agnfinder/tf_sampling/tensorrt.py
# RUN /opt/tensorrt/python/python_setup.sh