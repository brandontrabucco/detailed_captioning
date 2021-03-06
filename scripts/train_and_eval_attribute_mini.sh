#!/bin/bash

echo "Starting experiment on mini COCO dataset with attributes."

# First remove previous checkpoints

echo "Removing old checkpoints."

rm -rf ckpts/show_and_tell_attribute
rm -rf ckpts/show_attend_and_tell_attribute
rm -rf ckpts/spatial_attention_attribute
rm -rf ckpts/visual_sentinel_attribute
rm -rf ckpts/up_down_attribute

# Second train all models

echo "Training all models."

python scripts/train/attribute/train_show_and_tell_attribute.py --num_epochs 100 --batch_size 32 --is_mini
python scripts/train/attribute/train_show_attend_and_tell_attribute.py --num_epochs 100 --batch_size 32 --is_mini
python scripts/train/attribute/train_spatial_attention_attribute.py --num_epochs 100 --batch_size 32 --is_mini
python scripts/train/attribute/train_visual_sentinel_attribute.py --num_epochs 100 --batch_size 32 --is_mini
python scripts/train/attribute/train_up_down_attribute.py --num_epochs 100 --batch_size 32 --is_mini

# Third eval all models

echo "Validating all models on the training set."

python scripts/eval/attribute/eval_show_and_tell_attribute.py --is_mini --batch_size 32 --beam_size 3 --mode train
python scripts/eval/attribute/eval_show_attend_and_tell_attribute.py --is_mini --batch_size 32 --beam_size 3 --mode train
python scripts/eval/attribute/eval_spatial_attention_attribute.py --is_mini --batch_size 32 --beam_size 3 --mode train
python scripts/eval/attribute/eval_visual_sentinel_attribute.py --is_mini --batch_size 32 --beam_size 3 --mode train
python scripts/eval/attribute/eval_up_down_attribute.py --is_mini --batch_size 32 --beam_size 3 --mode train

python scripts/dataviz/plot_metrics.py --mode train

echo "Validating all models on the eval set."

python scripts/eval/attribute/eval_show_and_tell_attribute.py --is_mini --batch_size 32 --beam_size 3 --mode eval
python scripts/eval/attribute/eval_show_attend_and_tell_attribute.py --is_mini --batch_size 32 --beam_size 3 --mode eval
python scripts/eval/attribute/eval_spatial_attention_attribute.py --is_mini --batch_size 32 --beam_size 3 --mode eval
python scripts/eval/attribute/eval_visual_sentinel_attribute.py --is_mini --batch_size 32 --beam_size 3 --mode eval
python scripts/eval/attribute/eval_up_down_attribute.py --is_mini --batch_size 32 --beam_size 3 --mode eval


python scripts/dataviz/plot_metrics.py --mode eval

echo "Finished the image captioning experiment."