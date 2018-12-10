#!/bin/bash
python scripts/eval/eval_show_and_tell.py --is_mini --batch_size 32 --beam_size 3
python scripts/eval/eval_show_attend_and_tell.py --is_mini --batch_size 32 --beam_size 3
python scripts/eval/eval_spatial_attention.py --is_mini --batch_size 32 --beam_size 3
python scripts/eval/eval_visual_sentinel.py --is_mini --batch_size 32 --beam_size 3
python scripts/eval/eval_up_down.py --is_mini --batch_size 32 --beam_size 3