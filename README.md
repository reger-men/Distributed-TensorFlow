# Distributed-TensorFlow 
Specifying distributed devices in your model.
Train CNN Cifar10 or MNIST over multiple devices to demonstrate the model parallelism approch.
In this example we use only one machine with mutliple GPUs.

## Run 
1. ```./run.sh```.
2. ```CUDA_VISIBLE_DEVICES=0 python3 train.py``` That will create and run the Tensorflow graph.

## Reference
https://www.tensorflow.org/deploy/distributed
