#!/bin/bash
echo "Downloading model"
wget http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
tar -xzf ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
mv  -v ./ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/* ./
echo "Finished extracting model"
