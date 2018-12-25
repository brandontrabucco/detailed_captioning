#!/bin/bash
python scripts/train/train_show_and_tell.py --num_epochs 100 --batch_size 32 --is_mini
python scripts/train/train_show_attend_and_tell.py --num_epochs 100 --batch_size 32 --is_mini
python scripts/train/train_spatial_attention.py --num_epochs 100 --batch_size 32 --is_mini
python scripts/train/train_visual_sentinel.py --num_epochs 100 --batch_size 32 --is_mini
python scripts/train/train_up_down.py --num_epochs 100 --batch_size 32 --is_mini