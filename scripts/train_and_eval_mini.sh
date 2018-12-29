#!/bin/bash

echo "Starting experiment on mini COCO dataset."

# First remove previous checkpoints

echo "Removing old checkpoints."

rm -rf ckpts/show_and_tell
rm -rf ckpts/show_attend_and_tell
rm -rf ckpts/spatial_attention
rm -rf ckpts/visual_sentinel
rm -rf ckpts/up_down
rm -rf ckpts/up_down_part_of_speech

# Second train all models

echo "Training all models."

python scripts/train/train_show_and_tell.py --num_epochs 100 --batch_size 32 --is_mini
python scripts/train/train_show_attend_and_tell.py --num_epochs 100 --batch_size 32 --is_mini
python scripts/train/train_spatial_attention.py --num_epochs 100 --batch_size 32 --is_mini
python scripts/train/train_visual_sentinel.py --num_epochs 100 --batch_size 32 --is_mini
python scripts/train/train_up_down.py --num_epochs 100 --batch_size 32 --is_mini
python scripts/train/train_up_down_part_of_speech.py --num_epochs 100 --batch_size 32 --is_mini

# Third eval all models

echo "Validating all models on the training set."

python scripts/eval/eval_show_and_tell.py --is_mini --batch_size 32 --beam_size 3 --mode train
python scripts/eval/eval_show_attend_and_tell.py --is_mini --batch_size 32 --beam_size 3 --mode train
python scripts/eval/eval_spatial_attention.py --is_mini --batch_size 32 --beam_size 3 --mode train
python scripts/eval/eval_visual_sentinel.py --is_mini --batch_size 32 --beam_size 3 --mode train
python scripts/eval/eval_up_down.py --is_mini --batch_size 32 --beam_size 3 --mode train
python scripts/eval/eval_up_down_part_of_speech.py --is_mini --batch_size 32 --beam_size 3 --mode train

python scripts/dataviz/plot_metrics.py --mode train

echo "Validating all models on the eval set."

python scripts/eval/eval_show_and_tell.py --is_mini --batch_size 32 --beam_size 3 --mode eval
python scripts/eval/eval_show_attend_and_tell.py --is_mini --batch_size 32 --beam_size 3 --mode eval
python scripts/eval/eval_spatial_attention.py --is_mini --batch_size 32 --beam_size 3 --mode eval
python scripts/eval/eval_visual_sentinel.py --is_mini --batch_size 32 --beam_size 3 --mode eval
python scripts/eval/eval_up_down.py --is_mini --batch_size 32 --beam_size 3 --mode eval
python scripts/eval/eval_up_down_part_of_speech.py --is_mini --batch_size 32 --beam_size 3 --mode eval

python scripts/dataviz/plot_metrics.py --mode eval

echo "Finished the image captioning experiment."