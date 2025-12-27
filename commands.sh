
python -m a40.weight_calibration.main --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

python -m a40.activation_calibration.main --layers 0,1,2,3,4,5,6,7,9,10,11,12,13,14,15

python -m a40.main --train-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15


# Train on cpu
python -m a40.main --device cpu --train-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# Train on gpu
python -m a40.main --train-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

python -m a40.main_relu --no-compile

TOKENIZERS_PARALLELISM=false torchrun --standalone --nproc_per_node=8 -m a40.custom.main

TOKENIZERS_PARALLELISM=false python -m a40.custom.main

# Eval
python -m a40.eval --checkpoint-step 1000 --train-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --batch-size 16 --seq-len 512 --tasks gsm8k

python -m a40.eval --checkpoint-name "r2-408" --eval-include-teacher

python -m a40.eval \
  --checkpoint-name r2-408 \
  --include-teacher \
  --limit 0.01 \
  --tasks gsm8k mmlu hellaswag arc_challenge winogrande humaneval mbpp \
  --output eval_results_r2-408-core.json


# upload
rclone copy a40/checkpoints/student_final/r5-62/ \
  gdrive:checkpoints/r5-62/ \
  --progress


# download
rclone copy gdrive:checkpoints/r5-159 /workspace/a40/checkpoints/r5-159 --progress


