#!/bin/bash
echo "Downloading model"
mkdir ./ckpts/resnet_v2_101
wget http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz
tar -C ./ckpts/resnet_v2_101/ -xzf resnet_v2_101_2017_04_14.tar.gz
rm resnet_v2_101_2017_04_14.tar.gz
echo "Finished extracting model"
