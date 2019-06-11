FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04


# Get basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        wget \
        git \
        python3 \
        python3-dev \
        python3-pip \
        python3-wheel \
        python3-numpy \
        libcurl3-dev  \
        ca-certificates \
        gcc \
        sox \
        libsox-fmt-mp3 \
        htop \
        nano \
        swig \
        cmake \
        libboost-all-dev \
        zlib1g-dev \
        libbz2-dev \
        liblzma-dev \
        locales \
        pkg-config \
        libsox-dev \
        openjdk-8-jdk \
        bash-completion \
        g++ \
        unzip

RUN ln -s -f /usr/bin/python3 /usr/bin/python

# Install NCCL 2.2
RUN apt-get install -qq -y --allow-downgrades --allow-change-held-packages libnccl2=2.3.7-1+cuda10.0 libnccl-dev=2.3.7-1+cuda10.0

# Install Bazel
RUN curl -LO "https://github.com/bazelbuild/bazel/releases/download/0.19.2/bazel_0.19.2-linux-x86_64.deb"
RUN dpkg -i bazel_*.deb

# Install CUDA CLI Tools
RUN apt-get install -qq -y cuda-command-line-tools-10-0

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# << END Install base software


# Put cuda libraries to where they are expected to be
RUN mkdir /usr/local/cuda/lib &&  \
    ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/local/cuda/lib/libnccl.so.2 && \
    ln -s /usr/include/nccl.h /usr/local/cuda/include/nccl.h && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    ln -s /usr/include/cudnn.h /usr/local/cuda/include/cudnn.h


# Set library paths
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/:/usr/local/cuda/lib64/stubs/


# Allow Python printing utf-8
ENV PYTHONIOENCODING UTF-8


RUN pip3 install tensorflow-gpu==2.0.0-alpha0 pillow jupyter matplotlib


RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_notebook_config.py

# ========== Set entrypoint and command ==========
ENTRYPOINT ["/usr/local/bin/jupyter", "notebook", "--allow-root", "--ip=0.0.0.0"]
