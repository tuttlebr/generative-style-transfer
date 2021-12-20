FROM nvcr.io/nvidia/tensorflow:21.08-tf2-py3

ENV NB_USER jovyan
ENV NB_UID 1000
ENV NB_PREFIX /
ENV HOME /app

RUN useradd -M -s /bin/bash -N -u ${NB_UID} ${NB_USER} \
    && mkdir -p ${HOME} \
    && chown -R ${NB_USER}:users ${HOME} 

USER $NB_UID

WORKDIR ${HOME}

COPY style_transfer.py .

RUN wget -O model_top.h5 https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    && pip3 --no-cache-dir install --user \
    pillow \
    matplotlib \
    tqdm \
    tensorflow-addons

RUN mkdir results
