#!/bin/bash
xhost +
docker run -it  --ipc=host -p 5555:5555 -v ~/docker/shared:/docker/shared -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/root/.Xauthority --privileged --net=host --env="QT_X11_NO_MITSHM=1" udacity_drl jupyter-notebook --ip 0.0.0.0 --port 5555 --allow-root
