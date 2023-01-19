#! /bin/bash
BASE=$(dirname $(realpath $0))

docker run \
    --runtime=nvidia \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -v ${BASE}:/project/ \
    --network=host \
    -ti \
    --shm-size=14gb \
    localization bash
