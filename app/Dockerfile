FROM tensorflow/tensorflow:2.2.3-gpu-py3@sha256:0cf791ccf53a81debc86360bc7918f711f47b40cf0f451a748b256a47c4e1cf6

ARG EPOCHS=3
ARG STEPS_PER_EPOCH=200
ARG STYLE_WEIGHT=.002
ARG CONTENT_WEIGHT=10000
ARG TOTAL_VARIATION_WEIGHT=100000000
ARG MAX_DIM=512
ARG USER=brandon

ENV EPOCHS=${EPOCHS}
ENV STEPS_PER_EPOCH=${STEPS_PER_EPOCH}
ENV STYLE_WEIGHT=${STYLE_WEIGHT}
ENV CONTENT_WEIGHT=${CONTENT_WEIGHT}
ENV TOTAL_VARIATION_WEIGHT=${TOTAL_VARIATION_WEIGHT}
ENV MAX_DIM=${MAX_DIM}
ENV USER=${USER}
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home/${USER}

WORKDIR ${HOME}

RUN useradd -M -s /bin/bash -N -u 1000 ${USER} \
    && chown -R ${USER}:users /usr/local/bin \
    && chown -R ${USER}:users ${HOME}

COPY vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    /home/${USER}/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5

COPY style_transfer.py style_transfer.py

RUN pip3 --no-cache-dir install \
    pillow~=8.3.0 \
    matplotlib~=3.3.0 \
    tqdm~=4.61.0

USER ${USER}

RUN mkdir results

ENV TF_CPP_MIN_LOG_LEVEL=1

ENTRYPOINT python3 style_transfer.py \
    -c content.jpg \
    -s style.jpg \
    --epochs ${EPOCHS} \
    --steps_per_epoch ${STEPS_PER_EPOCH} \
    --style_weight ${STYLE_WEIGHT} \
    --content_weight ${CONTENT_WEIGHT} \
    --total_variation_weight ${TOTAL_VARIATION_WEIGHT} \
    --max_dim ${MAX_DIM}
