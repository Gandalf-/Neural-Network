#!/bin/bash

# Austin Voecks
# run.sh

run="   java -cp .:commons-math3-3.6.1.jar MNIST"
sample="data/mnist_train.csv 6000  data/mnist_test.csv 1000"
full="  data/mnist_train.csv 60000 data/mnist_test.csv 10000"

if [[ "$1" == "sample" ]]; then
  echo "[run.sh] Running on sample set"
  eval "$run" "$sample"
  exit
fi

echo "[run.sh] Running on full set"
eval "$run" "$full"
