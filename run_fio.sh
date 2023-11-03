#!/bin/bash

declare -a fio_commands=(
    "fio --name=read --size=1g --direct=0 --bs=1g --rw=read \
      --ioengine=psync --iodepth=1 --directory=/home/jack/temp/fio"
    "fio --name=read --size=1g --direct=1 --bs=128k --rw=read \
      --ioengine=io_uring --iodepth=32 --directory=/home/jack/temp/fio"
)

for cmd in "${fio_commands[@]}"
do   
    eval $cmd
    sleep 1
    eval $cmd
    sleep 1
done
