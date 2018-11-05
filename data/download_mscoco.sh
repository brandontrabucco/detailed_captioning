#!/bin/bash
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
mkdir ./data/coco
unzip train2017.zip -d ./data/coco
unzip val2017.zip -d ./data/coco
unzip test2017.zip -d ./data/coco
unzip annotations_trainval2017.zip -d ./data/coco
rm train2017.zip
rm val2017.zip
rm test2017.zip
rm annotations_trainval2017.zip