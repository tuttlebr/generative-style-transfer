FROM nvcr.io/nvidia/tensorflow:20.08-tf2-py3

ARG CONTENT_PATH=sample_images/daphnee_cat.jpg
ARG STYLE_PATH=sample_images/sample_images/the_fall_of_phaeton.jpg
ARG EPOCHS=3
ARG STEPS_PER_EPOCH=200
ARG STYLE_WEIGHT=.002
ARG CONTENT_WEIGHT=10000
ARG TOTAL_VARIATION_WEIGHT=100000000
ARG MAX_DIM=512
ARG USER=brandon

ENV CONTENT_PATH=${CONTENT_PATH}
ENV STYLE_PATH=${STYLE_PATH}
ENV EPOCHS=${EPOCHS}
ENV STEPS_PER_EPOCH=${STEPS_PER_EPOCH}
ENV STYLE_WEIGHT=${STYLE_WEIGHT}
ENV CONTENT_WEIGHT=${CONTENT_WEIGHT}
ENV TOTAL_VARIATION_WEIGHT=${TOTAL_VARIATION_WEIGHT}
ENV MAX_DIM=${MAX_DIM}
ENV USER=${USER}
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home/${USER}
ENV TF_CPP_MIN_LOG_LEVEL=1
WORKDIR ${HOME}

RUN useradd -M -s /bin/bash -N -u 1000 ${USER} \
    && chown -R ${USER}:users /usr/local/bin \
    && chown -R ${USER}:users ${HOME}

COPY vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    /home/${USER}/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
    
COPY style_transfer.py /home/${USER}/style_transfer.py
COPY ${CONTENT_PATH} /home/${USER}/content.jpg
COPY ${STYLE_PATH} /home/${USER}/style.jpg


RUN pip3 --no-cache-dir install \
    pillow \
    matplotlib \
    progressbar

USER ${USER}

RUN mkdir results

ENTRYPOINT python3 style_transfer.py \
    -c content.jpg \
    -s style.jpg \
    --epochs ${EPOCHS} \
    --steps_per_epoch ${STEPS_PER_EPOCH} \
    --style_weight ${STYLE_WEIGHT} \
    --content_weight ${CONTENT_WEIGHT} \
    --total_variation_weight ${TOTAL_VARIATION_WEIGHT} \
    --max_dim ${MAX_DIM}