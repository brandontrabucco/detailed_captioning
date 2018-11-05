#!/bin/bash
echo "Downloading model"
mkdir ./ckpts/faster_rcnn_resnet101_coco
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz
tar -xzf faster_rcnn_resnet101_coco_2018_01_28.tar.gz
mv  -v ./faster_rcnn_resnet101_coco_2018_01_28/* ./ckpts/faster_rcnn_resnet101_coco/
rm faster_rcnn_resnet101_coco_2018_01_28.tar.gz
rm -r ./faster_rcnn_resnet101_coco_2018_01_28
echo "Finished extracting model"

