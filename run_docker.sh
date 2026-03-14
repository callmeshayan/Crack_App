#!/bin/bash

# Build the Docker image if needed
docker build -t crack_app .

# Run with rpicam tools and libraries mounted from host
docker run --rm -it \
  --device=/dev/video0:/dev/video0 \
  --device=/dev/video1:/dev/video1 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /usr/bin/rpicam-vid:/usr/bin/rpicam-vid:ro \
  -v /usr/bin/rpicam-hello:/usr/bin/rpicam-hello:ro \
  -v /usr/bin/rpicam-jpeg:/usr/bin/rpicam-jpeg:ro \
  -v /usr/bin/rpicam-still:/usr/bin/rpicam-still:ro \
  -v /usr/bin/rpicam-raw:/usr/bin/rpicam-raw:ro \
  -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro \
  --privileged \
  crack_app
