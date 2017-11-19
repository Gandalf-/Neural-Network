#!/bin/bash

run="java -cp .:commons-math3-3.6.1.jar "
sample="data/mnist_train.csv 6000  data/mnist_test.csv 1000"
full="data/mnist_train.csv 60000 data/mnist_test.csv 10000"

h_sample="run on sample MNIST data set"
h_full="run on full MNIST data set"
h_sand="start the sandbox example program"

case $1 in
  sample)
    eval "$run MNIST $sample"
    ;;

  sandbox)
    eval "$run Sandbox"
    ;;

  full)
    eval "$run MNIST $full"
    ;;

  ''|*)
    echo "
options
  full      $h_full
  sample    $h_sample
  sandbox   $h_sand
  "
esac
