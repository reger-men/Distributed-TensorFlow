# Distributed-TensorFlow 
Specifying distributed devices in your model.
Train CNN Cifar10 over multiple devices to demonstrate the model parallelism approch.
In this example we use only one machine with mutliple GPUs.

## Run 
1. ```python task.py --task_index=1```.
2. ```python train.py``` That will create and run the Tensorflow graph.

## Reference
https://www.tensorflow.org/deploy/distributed
