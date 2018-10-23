#!/bin/bash
python preprocess_mscoco.py --train_image_dir './coco/train2017' --val_image_dir './coco/val2017' --train_captions_file './coco/annotations/captions_train2017.json' --val_captions_file './coco/annotations/captions_val2017.json' --output_dir './coco/' --vocab_size 100000 --embedding_size 300
