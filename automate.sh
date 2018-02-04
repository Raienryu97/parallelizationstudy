#!/bin/bash

cmake .
make clean
cd outputs && rm -rf *.jpg && cd ..
make -j8

echo ""
echo "Running application now"
echo ""

./lk
gprof lk gmon.out > analysis.txt
