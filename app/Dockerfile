FROM nvcr.io/nvidia/tensorflow:20.10-tf2-py3

WORKDIR /app

COPY style_transfer.py .

RUN pip3 --no-cache-dir install --user \
    pillow~=8.3.0 \
    matplotlib~=3.3.0 \
    tqdm~=4.61.0 \
    tensorflow-addons~=0.13.0

RUN mkdir results
