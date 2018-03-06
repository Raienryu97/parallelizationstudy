#!/bin/bash

echo "Compiling single core variant"
cc mat.c -o singlecore
echo "Compiing multi core variant"
cc mat.c -fopenmp -o multicore
echo "Compiling GPU accelerated variant"
pgcc mat.c -ta=tesla:cc50 -o gpuaccel

echo ""
echo ""
echo "Running single core variant"
./singlecore

echo ""
echo "Running multi core variant"
./multicore

echo ""
echo "Running GPU accelerated variant"
./gpuaccel
