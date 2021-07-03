FROM nvcr.io/nvidia/tensorflow:20.08-tf2-py3
ARG USER=brandon

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home/${USER}

RUN pip3 --no-cache-dir install pillow jupyter-lab matplotlib flask flask_restplus progressbar absl

RUN useradd -M -s /bin/bash -N -u 1000 ${USER} \
    && chown -R ${USER}:users /usr/local/bin \
    && chown -R ${USER}:users ${HOME}

USER ${USER}

CMD ["sh","-c", "jupyter lab --notebook-dir=${HOME} --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*'""]
