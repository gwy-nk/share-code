# caffe-kl

The code is based on Caffe for the problem of kl loss.

# Notes: 

-prepare a list with format like:
1.img 0.1 0.2 0.2 ... 
2.img 0.3 0.5 0.2 ...
[default distribution dimension is 8]
(if you need to change the dimension, change label_num in src/caffe/layers/data_layer.cpp)
-create lmdb use create_imagenet_r.sh

# Author:

Dongyu She
sherry6656@163.com
Nankai University, IMI.
