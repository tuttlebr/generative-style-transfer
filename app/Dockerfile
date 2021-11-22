FROM nvcr.io/nvidia/tensorflow:20.10-tf2-py3

ARG USER=jovyan

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
    tensorflow-addons~=0.13.0

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
