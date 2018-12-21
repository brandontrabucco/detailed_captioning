#!/bin/bash
python ./data/preextract_features_mscoco.py --train_image_dir './data/coco/train2017' --val_image_dir './data/coco/val2017' --train_captions_file './data/coco/annotations/captions_train2017.json' --val_captions_file './data/coco/annotations/captions_val2017.json' --output_dir './data/coco_micro/' --vocab_size 100000 --embedding_size 300 --train_dataset_size 256 --val_dataset_size 8 --test_dataset_size 8