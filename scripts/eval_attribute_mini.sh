#!/bin/bash

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