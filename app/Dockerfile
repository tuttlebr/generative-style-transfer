FROM tensorflow/tensorflow:2.3.3-gpu@sha256:62c0c4bac0f762e1f0c4ac537374a409a6aefa2236a4ef790c7dced411c67df0

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

COPY style_transfer.py style_transfer.py

USER ${USER}

RUN pip3 --no-cache-dir install --user\
    pillow~=8.3.0 \
    matplotlib~=3.3.0 \
    tqdm~=4.61.0 \
    tensorflow-hub~=0.12.0 \
    tensorflow-addons

RUN mkdir results

EXPOSE 8000

ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_XLA_FLAGS="--tf_xla_auto_jit=2"
ENV TF_ENABLE_AUTO_MIXED_PRECISION=1
ENV TFHUB_CACHE_DIR=results

ENTRYPOINT python3 style_transfer.py \
    -c content.jpg \
    -s style.jpg \
    --epochs ${EPOCHS} \
    --steps_per_epoch ${STEPS_PER_EPOCH} \
    --style_weight ${STYLE_WEIGHT} \
    --content_weight ${CONTENT_WEIGHT} \
    --total_variation_weight ${TOTAL_VARIATION_WEIGHT} \
    --max_dim ${MAX_DIM}
