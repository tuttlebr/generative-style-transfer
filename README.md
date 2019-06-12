# generative-style-transfer
Generative-style-transfer with tensorflow 2

A simple implementation of neural style transfer in python using Tensorflow 2.0.

<img src="https://github.com/tuttlebr/generative-style-transfer/blob/master/cat_content.jpg" width="100">
<img src="https://github.com/tuttlebr/generative-style-transfer/blob/master/the_fall_of_phaeton.jpg" width="100">

## Requirements
[Docker-ce](https://docs.docker.com/v17.12/install/ "Docker Installation Info")

[nvidia-docker2](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0) "Nvidia Docker Install Info")

[Nvidia Driver](https://www.nvidia.com/Download/index.aspx "Nvidia driver installation >= 410.*")

## Build docker Image

```bash
docker build -f Dockerfile -t generative_image .
```

## Build docker Container 
*with Jupyter Notebook on port http://localhost:8888*

```bash
nvidia-docker run -it -d \
  -p 8888:8888 \
  -u $(id -u):$(id -g) \
  -e HOME=/home/$USER \
  -v /home/$USER:/home/$USER \
  generative_image  \
  --notebook-dir=$PWD
``` 

## Run an example within container
```bash
docker exec -it {container_id} python3 StyleTransfer.py \
  -c cat_content.jpg \
  -s the_fall_of_phaeton.jpg \
  --epoch 3
```

