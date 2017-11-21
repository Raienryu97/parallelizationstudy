#!/bin/bash

# This script has been written to test the execution times
# of the serial and parallel variants of the LK Optical flow
# method. It automatically cleans and builds everytime it is run.


number_of_runs=100;
result_parallel=0;
result_serial=0;
sum_parallel=0;
sum_serial=0;

cmake .
make clean
make -j8

for((i=1;i<$number_of_runs;i++))
  do
    result_parallel=$(./lkp im1.jpg im2.jpg);
    sum_parallel=$(python -c "print $sum_parallel+$result_parallel");
    result_serial=$(./lks im1.jpg im2.jpg);
    sum_serial=$(python -c "print $sum_serial+$result_serial");
  done

avg_parallel=$(python -c "print $sum_parallel/$number_of_runs");
avg_serial=$(python -c "print $sum_serial/$number_of_runs");
improvement=$(python -c "print (($avg_serial-$avg_parallel)/$avg_serial)*100");

echo "";
echo "Average of $number_of_runs code executions $avg_serial seconds";
echo "Average of $number_of_runs OpenMP optimised code executions $avg_parallel seconds";
echo "Improvement: $improvement %";
echo "";
